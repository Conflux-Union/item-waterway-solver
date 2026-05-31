use mc_schem::region::WorldSlice;
use mc_schem::{Block, Region, Schematic};
use std::collections::BTreeSet;
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CollisionKind {
    NonSolid,
    FullBlock,
    PartialBlock,
    UnsupportedPartial,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct CollisionBox {
    pub min_x: f64,
    pub min_y: f64,
    pub min_z: f64,
    pub max_x: f64,
    pub max_y: f64,
    pub max_z: f64,
}

impl CollisionBox {
    const FULL_BLOCK: Self = Self {
        min_x: 0.0,
        min_y: 0.0,
        min_z: 0.0,
        max_x: 1.0,
        max_y: 1.0,
        max_z: 1.0,
    };
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct CollisionBoxes {
    boxes: [CollisionBox; 4],
    count: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FaceDirection {
    North,
    South,
    West,
    East,
    Down,
    Up,
}

impl CollisionBoxes {
    fn empty() -> Self {
        Self::default()
    }

    fn single(collision_box: CollisionBox) -> Self {
        let mut result = Self::empty();
        result.push(collision_box);
        result
    }

    fn push(&mut self, collision_box: CollisionBox) {
        if self.count < self.boxes.len() {
            self.boxes[self.count] = collision_box;
            self.count += 1;
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.count == 0
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = &CollisionBox> {
        self.boxes[..self.count].iter()
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct WaterCell {
    pub height: f64,
    pub own_height: f64,
    pub falling: bool,
}

#[derive(Clone, Debug)]
pub(crate) struct LoadedSchematic {
    pub name: String,
    pub region: Region,
    pub approximate_collision_blocks: Vec<String>,
}

pub(crate) fn load_litematic(path: &Path) -> Result<LoadedSchematic, String> {
    let filename = path
        .to_str()
        .ok_or_else(|| format!("Projection path is not valid UTF-8: {}", path.display()))?;
    let (schematic, _) = Schematic::from_file(filename)
        .map_err(|error| format!("Failed to read litematic {}: {error}", path.display()))?;
    let region = schematic.to_single_region(&Block::air());
    let approximate_collision_blocks = collect_approximate_collision_blocks(&region);
    Ok(LoadedSchematic {
        name: path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or("projection")
            .to_string(),
        region,
        approximate_collision_blocks,
    })
}

pub(crate) fn block_full_id(block: &Block) -> String {
    block.full_id()
}

pub(crate) fn block_at<'a>(region: &'a Region, pos: [i32; 3]) -> Option<&'a Block> {
    region.block_at(pos)
}

pub(crate) fn water_at(region: &Region, pos: [i32; 3]) -> Option<WaterCell> {
    let block = region.block_at(pos)?;
    let raw = raw_water_cell(block)?;
    let height = if has_water_above(region, pos) {
        1.0
    } else {
        raw.own_height
    };
    Some(WaterCell {
        height,
        own_height: raw.own_height,
        falling: raw.falling,
    })
}

pub(crate) fn collision_kind(block: &Block) -> CollisionKind {
    if block.is_air() || is_water_block(block) {
        return CollisionKind::NonSolid;
    }
    if block.namespace != "minecraft" {
        return CollisionKind::FullBlock;
    }
    if is_non_solid_block_id(&block.id) {
        return CollisionKind::NonSolid;
    }
    if let Some(kind) = supported_partial_collision_kind(block) {
        return kind;
    }
    if is_partial_collision_block_id(&block.id) {
        return CollisionKind::UnsupportedPartial;
    }
    CollisionKind::FullBlock
}

pub(crate) fn collision_boxes(block: &Block) -> CollisionBoxes {
    match collision_kind(block) {
        CollisionKind::NonSolid | CollisionKind::UnsupportedPartial => CollisionBoxes::empty(),
        CollisionKind::FullBlock => CollisionBoxes::single(CollisionBox::FULL_BLOCK),
        CollisionKind::PartialBlock => supported_partial_collision_boxes(block),
    }
}

pub(crate) fn is_collision_shape_full_block(block: &Block) -> bool {
    matches!(collision_kind(block), CollisionKind::FullBlock)
}

pub(crate) fn merged_face_occludes(
    source: &Block,
    target: &Block,
    direction: FaceDirection,
) -> bool {
    let mut rectangles = face_rectangles(&collision_boxes(source), direction);
    rectangles.extend(face_rectangles(
        &collision_boxes(target),
        direction.opposite(),
    ));
    rectangles_cover_unit_square(&rectangles)
}

pub(crate) fn is_face_sturdy(block: &Block, direction: FaceDirection) -> bool {
    if block.is_air() || is_water_block(block) || is_ice(block) {
        return false;
    }

    match collision_kind(block) {
        CollisionKind::NonSolid | CollisionKind::UnsupportedPartial => false,
        CollisionKind::FullBlock => true,
        CollisionKind::PartialBlock => face_is_full(&collision_boxes(block), direction),
    }
}

pub(crate) fn blocks_motion(block: &Block) -> bool {
    !collision_boxes(block).is_empty()
}

fn face_is_full(boxes: &CollisionBoxes, direction: FaceDirection) -> bool {
    rectangles_cover_unit_square(&face_rectangles(boxes, direction))
}

fn face_rectangles(boxes: &CollisionBoxes, direction: FaceDirection) -> Vec<(f64, f64, f64, f64)> {
    let mut rectangles = Vec::with_capacity(4);
    for collision_box in boxes.iter() {
        let rectangle = match direction {
            FaceDirection::North if approx_eq(collision_box.min_z, 0.0) => Some((
                collision_box.min_x,
                collision_box.max_x,
                collision_box.min_y,
                collision_box.max_y,
            )),
            FaceDirection::South if approx_eq(collision_box.max_z, 1.0) => Some((
                collision_box.min_x,
                collision_box.max_x,
                collision_box.min_y,
                collision_box.max_y,
            )),
            FaceDirection::West if approx_eq(collision_box.min_x, 0.0) => Some((
                collision_box.min_z,
                collision_box.max_z,
                collision_box.min_y,
                collision_box.max_y,
            )),
            FaceDirection::East if approx_eq(collision_box.max_x, 1.0) => Some((
                collision_box.min_z,
                collision_box.max_z,
                collision_box.min_y,
                collision_box.max_y,
            )),
            FaceDirection::Down if approx_eq(collision_box.min_y, 0.0) => Some((
                collision_box.min_x,
                collision_box.max_x,
                collision_box.min_z,
                collision_box.max_z,
            )),
            FaceDirection::Up if approx_eq(collision_box.max_y, 1.0) => Some((
                collision_box.min_x,
                collision_box.max_x,
                collision_box.min_z,
                collision_box.max_z,
            )),
            _ => None,
        };
        if let Some(rectangle) = rectangle {
            rectangles.push(rectangle);
        }
    }
    rectangles
}

fn rectangles_cover_unit_square(rectangles: &[(f64, f64, f64, f64)]) -> bool {
    if rectangles.is_empty() {
        return false;
    }

    let mut u_boundaries = Vec::with_capacity(rectangles.len() * 2 + 2);
    let mut v_boundaries = Vec::with_capacity(rectangles.len() * 2 + 2);
    u_boundaries.extend([0.0, 1.0]);
    v_boundaries.extend([0.0, 1.0]);

    for &(min_u, max_u, min_v, max_v) in rectangles {
        u_boundaries.push(min_u.clamp(0.0, 1.0));
        u_boundaries.push(max_u.clamp(0.0, 1.0));
        v_boundaries.push(min_v.clamp(0.0, 1.0));
        v_boundaries.push(max_v.clamp(0.0, 1.0));
    }

    sort_and_dedup_boundaries(&mut u_boundaries);
    sort_and_dedup_boundaries(&mut v_boundaries);

    for u_window in u_boundaries.windows(2) {
        let min_u = u_window[0];
        let max_u = u_window[1];
        if max_u - min_u <= 1.0e-9 {
            continue;
        }
        let sample_u = (min_u + max_u) * 0.5;
        for v_window in v_boundaries.windows(2) {
            let min_v = v_window[0];
            let max_v = v_window[1];
            if max_v - min_v <= 1.0e-9 {
                continue;
            }
            let sample_v = (min_v + max_v) * 0.5;
            if !rectangles
                .iter()
                .any(|&(rect_min_u, rect_max_u, rect_min_v, rect_max_v)| {
                    sample_u > rect_min_u - 1.0e-9
                        && sample_u < rect_max_u + 1.0e-9
                        && sample_v > rect_min_v - 1.0e-9
                        && sample_v < rect_max_v + 1.0e-9
                })
            {
                return false;
            }
        }
    }

    true
}

fn sort_and_dedup_boundaries(values: &mut Vec<f64>) {
    values.sort_by(|left, right| left.total_cmp(right));
    values.dedup_by(|left, right| (*left - *right).abs() <= 1.0e-9);
}

fn approx_eq(left: f64, right: f64) -> bool {
    (left - right).abs() <= 1.0e-9
}

impl FaceDirection {
    fn opposite(self) -> Self {
        match self {
            Self::North => Self::South,
            Self::South => Self::North,
            Self::West => Self::East,
            Self::East => Self::West,
            Self::Down => Self::Up,
            Self::Up => Self::Down,
        }
    }
}

pub(crate) fn is_ice(block: &Block) -> bool {
    block.namespace == "minecraft"
        && matches!(
            block.id.as_str(),
            "ice" | "packed_ice" | "frosted_ice" | "blue_ice"
        )
}

fn collect_approximate_collision_blocks(region: &Region) -> Vec<String> {
    let mut ids = BTreeSet::new();
    for block in &region.palette {
        if matches!(collision_kind(block), CollisionKind::UnsupportedPartial) {
            ids.insert(block.full_id());
        }
    }
    ids.into_iter().collect()
}

fn has_water_above(region: &Region, pos: [i32; 3]) -> bool {
    region
        .block_at([pos[0], pos[1] + 1, pos[2]])
        .and_then(raw_water_cell)
        .is_some()
}

fn is_water_block(block: &Block) -> bool {
    raw_water_cell(block).is_some()
}

fn raw_water_cell(block: &Block) -> Option<WaterCell> {
    if block.namespace == "minecraft" && block.id == "water" {
        let level = block
            .attributes
            .get("level")
            .and_then(|value| value.parse::<u8>().ok())
            .unwrap_or(0);
        let (amount, falling) = if level == 0 {
            (8_u8, false)
        } else if level < 8 {
            (8_u8.saturating_sub(level), false)
        } else {
            (16_u8.saturating_sub(level), true)
        };
        let own_height = (amount as f32 / 9.0_f32) as f64;
        return Some(WaterCell {
            height: own_height,
            own_height,
            falling,
        });
    }

    if block
        .attributes
        .get("waterlogged")
        .map(|value| value == "true")
        .unwrap_or(false)
    {
        return Some(WaterCell {
            height: 1.0,
            own_height: (8.0_f32 / 9.0_f32) as f64,
            falling: false,
        });
    }

    None
}

fn supported_partial_collision_kind(block: &Block) -> Option<CollisionKind> {
    if block.id.ends_with("slab") {
        let slab_type = block
            .attributes
            .get("type")
            .map(String::as_str)
            .unwrap_or("bottom");
        return Some(if slab_type == "double" {
            CollisionKind::FullBlock
        } else {
            CollisionKind::PartialBlock
        });
    }
    if block.id.ends_with("trapdoor")
        || block.id.ends_with("door")
        || block.id.ends_with("stairs")
        || block.id.ends_with("pane")
        || block.id.ends_with("bars")
        || block.id.ends_with("wall")
        || block.id.ends_with("fence")
    {
        return Some(CollisionKind::PartialBlock);
    }
    if block.id.ends_with("fence_gate") {
        return Some(if bool_attr(block, "open") {
            CollisionKind::NonSolid
        } else {
            CollisionKind::PartialBlock
        });
    }
    None
}

fn supported_partial_collision_boxes(block: &Block) -> CollisionBoxes {
    if block.id.ends_with("slab") {
        return slab_collision_boxes(block);
    }
    if block.id.ends_with("trapdoor") {
        return trapdoor_collision_boxes(block);
    }
    if block.id.ends_with("fence_gate") {
        return fence_gate_collision_boxes(block);
    }
    if block.id.ends_with("door") {
        return door_collision_boxes(block);
    }
    if block.id.ends_with("stairs") {
        return stair_collision_boxes(block);
    }
    if block.id.ends_with("pane") || block.id.ends_with("bars") {
        return pane_collision_boxes(block);
    }
    if block.id.ends_with("wall") {
        return wall_collision_boxes(block);
    }
    if block.id.ends_with("fence") {
        return fence_collision_boxes(block);
    }
    CollisionBoxes::empty()
}

fn slab_collision_boxes(block: &Block) -> CollisionBoxes {
    match block
        .attributes
        .get("type")
        .map(String::as_str)
        .unwrap_or("bottom")
    {
        "top" => CollisionBoxes::single(CollisionBox {
            min_x: 0.0,
            min_y: 0.5,
            min_z: 0.0,
            max_x: 1.0,
            max_y: 1.0,
            max_z: 1.0,
        }),
        "double" => CollisionBoxes::single(CollisionBox::FULL_BLOCK),
        _ => CollisionBoxes::single(CollisionBox {
            min_x: 0.0,
            min_y: 0.0,
            min_z: 0.0,
            max_x: 1.0,
            max_y: 0.5,
            max_z: 1.0,
        }),
    }
}

fn trapdoor_collision_boxes(block: &Block) -> CollisionBoxes {
    const THICKNESS: f64 = 3.0 / 16.0;
    let facing = horizontal_direction(
        block
            .attributes
            .get("facing")
            .map(String::as_str)
            .unwrap_or("north"),
    );
    if bool_attr(block, "open") {
        return CollisionBoxes::single(rotate_y_clockwise(
            CollisionBox {
                min_x: 0.0,
                min_y: 0.0,
                min_z: 1.0 - THICKNESS,
                max_x: 1.0,
                max_y: 1.0,
                max_z: 1.0,
            },
            horizontal_turns(facing),
        ));
    }

    let is_top = block
        .attributes
        .get("half")
        .map(String::as_str)
        .unwrap_or("bottom")
        == "top";
    CollisionBoxes::single(CollisionBox {
        min_x: 0.0,
        min_y: if is_top { 1.0 - THICKNESS } else { 0.0 },
        min_z: 0.0,
        max_x: 1.0,
        max_y: if is_top { 1.0 } else { THICKNESS },
        max_z: 1.0,
    })
}

fn fence_gate_collision_boxes(block: &Block) -> CollisionBoxes {
    if bool_attr(block, "open") {
        return CollisionBoxes::empty();
    }

    let facing = horizontal_direction(
        block
            .attributes
            .get("facing")
            .map(String::as_str)
            .unwrap_or("north"),
    );
    if horizontal_axis_is_x(facing) {
        CollisionBoxes::single(CollisionBox {
            min_x: 6.0 / 16.0,
            min_y: 0.0,
            min_z: 0.0,
            max_x: 10.0 / 16.0,
            max_y: 1.5,
            max_z: 1.0,
        })
    } else {
        CollisionBoxes::single(CollisionBox {
            min_x: 0.0,
            min_y: 0.0,
            min_z: 6.0 / 16.0,
            max_x: 1.0,
            max_y: 1.5,
            max_z: 10.0 / 16.0,
        })
    }
}

fn door_collision_boxes(block: &Block) -> CollisionBoxes {
    const THICKNESS: f64 = 3.0 / 16.0;
    let facing = horizontal_direction(
        block
            .attributes
            .get("facing")
            .map(String::as_str)
            .unwrap_or("north"),
    );
    let door_direction = if bool_attr(block, "open") {
        if block
            .attributes
            .get("hinge")
            .map(String::as_str)
            .unwrap_or("left")
            == "right"
        {
            rotate_counter_clockwise(facing)
        } else {
            rotate_clockwise(facing)
        }
    } else {
        facing
    };

    CollisionBoxes::single(rotate_y_clockwise(
        CollisionBox {
            min_x: 0.0,
            min_y: 0.0,
            min_z: 1.0 - THICKNESS,
            max_x: 1.0,
            max_y: 1.0,
            max_z: 1.0,
        },
        horizontal_turns(door_direction),
    ))
}

fn stair_collision_boxes(block: &Block) -> CollisionBoxes {
    let facing = horizontal_direction(
        block
            .attributes
            .get("facing")
            .map(String::as_str)
            .unwrap_or("north"),
    );
    let shape = block
        .attributes
        .get("shape")
        .map(String::as_str)
        .unwrap_or("straight");
    let is_top = block.attributes.get("half").map(String::as_str) == Some("top");
    let (base_min_y, base_max_y, step_min_y, step_max_y) = if is_top {
        (0.5, 1.0, 0.0, 0.5)
    } else {
        (0.0, 0.5, 0.5, 1.0)
    };

    let mut result = CollisionBoxes::empty();
    result.push(CollisionBox {
        min_x: 0.0,
        min_y: base_min_y,
        min_z: 0.0,
        max_x: 1.0,
        max_y: base_max_y,
        max_z: 1.0,
    });

    match shape {
        "outer_left" => result.push(intersect_boxes(
            directional_half_box(facing, step_min_y, step_max_y),
            directional_half_box(rotate_counter_clockwise(facing), step_min_y, step_max_y),
        )),
        "outer_right" => result.push(intersect_boxes(
            directional_half_box(facing, step_min_y, step_max_y),
            directional_half_box(rotate_clockwise(facing), step_min_y, step_max_y),
        )),
        "inner_left" => {
            result.push(directional_half_box(facing, step_min_y, step_max_y));
            result.push(directional_half_box(
                rotate_counter_clockwise(facing),
                step_min_y,
                step_max_y,
            ));
        }
        "inner_right" => {
            result.push(directional_half_box(facing, step_min_y, step_max_y));
            result.push(directional_half_box(
                rotate_clockwise(facing),
                step_min_y,
                step_max_y,
            ));
        }
        _ => result.push(directional_half_box(facing, step_min_y, step_max_y)),
    }
    result
}

fn pane_collision_boxes(block: &Block) -> CollisionBoxes {
    cross_collision_boxes(
        2.0 / 16.0,
        1.0,
        2.0 / 16.0,
        1.0,
        8.0 / 16.0,
        true,
        bool_attr(block, "north"),
        bool_attr(block, "east"),
        bool_attr(block, "south"),
        bool_attr(block, "west"),
    )
}

fn fence_collision_boxes(block: &Block) -> CollisionBoxes {
    cross_collision_boxes(
        4.0 / 16.0,
        1.5,
        4.0 / 16.0,
        1.5,
        8.0 / 16.0,
        true,
        bool_attr(block, "north"),
        bool_attr(block, "east"),
        bool_attr(block, "south"),
        bool_attr(block, "west"),
    )
}

fn wall_collision_boxes(block: &Block) -> CollisionBoxes {
    cross_collision_boxes(
        8.0 / 16.0,
        1.5,
        6.0 / 16.0,
        1.5,
        11.0 / 16.0,
        bool_attr(block, "up"),
        wall_side_connected(block, "north"),
        wall_side_connected(block, "east"),
        wall_side_connected(block, "south"),
        wall_side_connected(block, "west"),
    )
}

fn cross_collision_boxes(
    post_width: f64,
    post_height: f64,
    arm_width: f64,
    arm_height: f64,
    arm_reach: f64,
    include_post: bool,
    north: bool,
    east: bool,
    south: bool,
    west: bool,
) -> CollisionBoxes {
    let mut result = CollisionBoxes::empty();
    if include_post {
        result.push(centered_column_box(post_width, post_height));
    }
    if north || south {
        result.push(CollisionBox {
            min_x: 0.5 - arm_width * 0.5,
            min_y: 0.0,
            min_z: if north { 0.0 } else { 1.0 - arm_reach },
            max_x: 0.5 + arm_width * 0.5,
            max_y: arm_height,
            max_z: if south { 1.0 } else { arm_reach },
        });
    }
    if east || west {
        result.push(CollisionBox {
            min_x: if west { 0.0 } else { 1.0 - arm_reach },
            min_y: 0.0,
            min_z: 0.5 - arm_width * 0.5,
            max_x: if east { 1.0 } else { arm_reach },
            max_y: arm_height,
            max_z: 0.5 + arm_width * 0.5,
        });
    }
    result
}

fn centered_column_box(width: f64, height: f64) -> CollisionBox {
    CollisionBox {
        min_x: 0.5 - width * 0.5,
        min_y: 0.0,
        min_z: 0.5 - width * 0.5,
        max_x: 0.5 + width * 0.5,
        max_y: height,
        max_z: 0.5 + width * 0.5,
    }
}

fn directional_half_box(direction: HorizontalDirection, min_y: f64, max_y: f64) -> CollisionBox {
    match direction {
        HorizontalDirection::North => CollisionBox {
            min_x: 0.0,
            min_y,
            min_z: 0.0,
            max_x: 1.0,
            max_y,
            max_z: 0.5,
        },
        HorizontalDirection::East => CollisionBox {
            min_x: 0.5,
            min_y,
            min_z: 0.0,
            max_x: 1.0,
            max_y,
            max_z: 1.0,
        },
        HorizontalDirection::South => CollisionBox {
            min_x: 0.0,
            min_y,
            min_z: 0.5,
            max_x: 1.0,
            max_y,
            max_z: 1.0,
        },
        HorizontalDirection::West => CollisionBox {
            min_x: 0.0,
            min_y,
            min_z: 0.0,
            max_x: 0.5,
            max_y,
            max_z: 1.0,
        },
    }
}

fn intersect_boxes(first: CollisionBox, second: CollisionBox) -> CollisionBox {
    CollisionBox {
        min_x: first.min_x.max(second.min_x),
        min_y: first.min_y.max(second.min_y),
        min_z: first.min_z.max(second.min_z),
        max_x: first.max_x.min(second.max_x),
        max_y: first.max_y.min(second.max_y),
        max_z: first.max_z.min(second.max_z),
    }
}

fn wall_side_connected(block: &Block, key: &str) -> bool {
    block
        .attributes
        .get(key)
        .map(String::as_str)
        .unwrap_or("none")
        != "none"
}

fn bool_attr(block: &Block, key: &str) -> bool {
    block.attributes.get(key).map(String::as_str) == Some("true")
}

#[derive(Clone, Copy)]
enum HorizontalDirection {
    North,
    East,
    South,
    West,
}

fn horizontal_direction(value: &str) -> HorizontalDirection {
    match value {
        "east" => HorizontalDirection::East,
        "south" => HorizontalDirection::South,
        "west" => HorizontalDirection::West,
        _ => HorizontalDirection::North,
    }
}

fn horizontal_turns(direction: HorizontalDirection) -> usize {
    match direction {
        HorizontalDirection::North => 0,
        HorizontalDirection::East => 1,
        HorizontalDirection::South => 2,
        HorizontalDirection::West => 3,
    }
}

fn rotate_clockwise(direction: HorizontalDirection) -> HorizontalDirection {
    match direction {
        HorizontalDirection::North => HorizontalDirection::East,
        HorizontalDirection::East => HorizontalDirection::South,
        HorizontalDirection::South => HorizontalDirection::West,
        HorizontalDirection::West => HorizontalDirection::North,
    }
}

fn rotate_counter_clockwise(direction: HorizontalDirection) -> HorizontalDirection {
    match direction {
        HorizontalDirection::North => HorizontalDirection::West,
        HorizontalDirection::West => HorizontalDirection::South,
        HorizontalDirection::South => HorizontalDirection::East,
        HorizontalDirection::East => HorizontalDirection::North,
    }
}

fn horizontal_axis_is_x(direction: HorizontalDirection) -> bool {
    matches!(
        direction,
        HorizontalDirection::East | HorizontalDirection::West
    )
}

fn rotate_y_clockwise(mut collision_box: CollisionBox, turns: usize) -> CollisionBox {
    for _ in 0..(turns % 4) {
        collision_box = CollisionBox {
            min_x: 1.0 - collision_box.max_z,
            min_y: collision_box.min_y,
            min_z: collision_box.min_x,
            max_x: 1.0 - collision_box.min_z,
            max_y: collision_box.max_y,
            max_z: collision_box.max_x,
        };
    }
    collision_box
}

fn is_non_solid_block_id(id: &str) -> bool {
    matches!(
        id,
        "air"
            | "cave_air"
            | "void_air"
            | "water"
            | "lava"
            | "bubble_column"
            | "glow_lichen"
            | "vine"
            | "weeping_vines"
            | "weeping_vines_plant"
            | "twisting_vines"
            | "twisting_vines_plant"
            | "seagrass"
            | "tall_seagrass"
            | "kelp"
            | "kelp_plant"
            | "grass"
            | "tall_grass"
            | "fern"
            | "large_fern"
            | "dead_bush"
            | "tripwire"
            | "redstone_wire"
            | "nether_portal"
            | "end_gateway"
            | "end_portal"
            | "structure_void"
            | "light"
            | "fire"
            | "soul_fire"
            | "torch"
            | "wall_torch"
            | "redstone_torch"
            | "redstone_wall_torch"
    )
}

fn is_partial_collision_block_id(id: &str) -> bool {
    id.contains("sign")
        || id.contains("banner")
        || id.contains("button")
        || id.contains("pressure_plate")
        || id.contains("trapdoor")
        || id.contains("door")
        || id.contains("fence")
        || id.contains("wall")
        || id.contains("pane")
        || id.contains("slab")
        || id.contains("stairs")
        || id.contains("rail")
        || id.contains("candle")
        || id.contains("cactus")
        || id.contains("chain")
        || id.contains("rod")
        || id.contains("ladder")
        || id.contains("skull")
        || id.contains("head")
        || id.contains("coral")
        || id.contains("amethyst")
        || id.contains("campfire")
        || id.contains("carpet")
        || id.contains("snow")
        || matches!(
            id,
            "lever"
                | "hopper"
                | "anvil"
                | "cauldron"
                | "bell"
                | "lectern"
                | "composter"
                | "lightning_rod"
                | "end_rod"
                | "chest"
                | "trapped_chest"
                | "barrel"
                | "decorated_pot"
        )
}
