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
    boxes: [CollisionBox; 12],
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
pub(crate) struct FluidCell {
    pub height: f64,
    pub own_height: f64,
    pub falling: bool,
}

pub(crate) type WaterCell = FluidCell;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FluidKind {
    Water,
    Lava,
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

pub(crate) fn fluid_at(region: &Region, pos: [i32; 3], fluid_kind: FluidKind) -> Option<FluidCell> {
    let block = region.block_at(pos)?;
    let raw = raw_fluid_cell(block, fluid_kind)?;
    let height = if has_fluid_above(region, pos, fluid_kind) {
        1.0
    } else {
        raw.own_height
    };
    Some(FluidCell {
        height,
        own_height: raw.own_height,
        falling: raw.falling,
    })
}

pub(crate) fn water_at(region: &Region, pos: [i32; 3]) -> Option<FluidCell> {
    fluid_at(region, pos, FluidKind::Water)
}

pub(crate) fn lava_at(region: &Region, pos: [i32; 3]) -> Option<FluidCell> {
    fluid_at(region, pos, FluidKind::Lava)
}

pub(crate) fn collision_kind(block: &Block) -> CollisionKind {
    if block.is_air() || is_intrinsic_fluid_block(block) {
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

fn support_boxes(block: &Block) -> CollisionBoxes {
    if block.namespace != "minecraft" {
        return collision_boxes(block);
    }

    match block.id.as_str() {
        "soul_sand" | "mud" => CollisionBoxes::single(CollisionBox::FULL_BLOCK),
        "snow" => snow_support_boxes(block),
        _ => collision_boxes(block),
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
    if block.is_air() || is_intrinsic_fluid_block(block) || is_ice(block) {
        return false;
    }

    match collision_kind(block) {
        CollisionKind::NonSolid | CollisionKind::UnsupportedPartial => false,
        CollisionKind::FullBlock => true,
        CollisionKind::PartialBlock => face_is_full(&support_boxes(block), direction),
    }
}

pub(crate) fn blocks_motion(block: &Block) -> bool {
    if block.namespace == "minecraft" && matches!(block.id.as_str(), "cobweb" | "bamboo_sapling") {
        return false;
    }

    if block.namespace == "minecraft" && block.id == "scaffolding" {
        return false;
    }

    let boxes = collision_boxes(block);
    if boxes.is_empty() {
        return false;
    }

    let Some(bounds) = collision_box_bounds(&boxes) else {
        return false;
    };
    let x_size = bounds.max_x - bounds.min_x;
    let y_size = bounds.max_y - bounds.min_y;
    let z_size = bounds.max_z - bounds.min_z;
    (x_size + y_size + z_size) / 3.0 >= 0.729_166_666_666_666_6 || y_size >= 1.0
}

fn collision_box_bounds(boxes: &CollisionBoxes) -> Option<CollisionBox> {
    let mut iter = boxes.iter().copied();
    let first = iter.next()?;
    let mut bounds = first;
    for collision_box in iter {
        bounds.min_x = bounds.min_x.min(collision_box.min_x);
        bounds.min_y = bounds.min_y.min(collision_box.min_y);
        bounds.min_z = bounds.min_z.min(collision_box.min_z);
        bounds.max_x = bounds.max_x.max(collision_box.max_x);
        bounds.max_y = bounds.max_y.max(collision_box.max_y);
        bounds.max_z = bounds.max_z.max(collision_box.max_z);
    }
    Some(bounds)
}

fn face_is_full(boxes: &CollisionBoxes, direction: FaceDirection) -> bool {
    rectangles_cover_unit_square(&face_rectangles(boxes, direction))
}

fn face_rectangles(boxes: &CollisionBoxes, direction: FaceDirection) -> Vec<(f64, f64, f64, f64)> {
    let Some(face_plane) = face_plane_coordinate(boxes, direction) else {
        return Vec::new();
    };
    let mut rectangles = Vec::with_capacity(4);
    for collision_box in boxes.iter() {
        let rectangle = match direction {
            FaceDirection::North if approx_eq(collision_box.min_z, face_plane) => Some((
                collision_box.min_x,
                collision_box.max_x,
                collision_box.min_y,
                collision_box.max_y,
            )),
            FaceDirection::South if approx_eq(collision_box.max_z, face_plane) => Some((
                collision_box.min_x,
                collision_box.max_x,
                collision_box.min_y,
                collision_box.max_y,
            )),
            FaceDirection::West if approx_eq(collision_box.min_x, face_plane) => Some((
                collision_box.min_z,
                collision_box.max_z,
                collision_box.min_y,
                collision_box.max_y,
            )),
            FaceDirection::East if approx_eq(collision_box.max_x, face_plane) => Some((
                collision_box.min_z,
                collision_box.max_z,
                collision_box.min_y,
                collision_box.max_y,
            )),
            FaceDirection::Down if approx_eq(collision_box.min_y, face_plane) => Some((
                collision_box.min_x,
                collision_box.max_x,
                collision_box.min_z,
                collision_box.max_z,
            )),
            FaceDirection::Up if approx_eq(collision_box.max_y, face_plane) => Some((
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

fn face_plane_coordinate(boxes: &CollisionBoxes, direction: FaceDirection) -> Option<f64> {
    let mut iter = boxes.iter().copied();
    let first = iter.next()?;
    let mut face_plane = match direction {
        FaceDirection::North => first.min_z,
        FaceDirection::South => first.max_z,
        FaceDirection::West => first.min_x,
        FaceDirection::East => first.max_x,
        FaceDirection::Down => first.min_y,
        FaceDirection::Up => first.max_y,
    };
    for collision_box in iter {
        face_plane = match direction {
            FaceDirection::North => face_plane.min(collision_box.min_z),
            FaceDirection::South => face_plane.max(collision_box.max_z),
            FaceDirection::West => face_plane.min(collision_box.min_x),
            FaceDirection::East => face_plane.max(collision_box.max_x),
            FaceDirection::Down => face_plane.min(collision_box.min_y),
            FaceDirection::Up => face_plane.max(collision_box.max_y),
        };
    }
    Some(face_plane)
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

fn has_fluid_above(region: &Region, pos: [i32; 3], fluid_kind: FluidKind) -> bool {
    region
        .block_at([pos[0], pos[1] + 1, pos[2]])
        .and_then(|block| raw_fluid_cell(block, fluid_kind))
        .is_some()
}

fn is_intrinsic_water_block(block: &Block) -> bool {
    block.namespace == "minecraft" && matches!(block.id.as_str(), "water" | "bubble_column")
}

fn is_intrinsic_lava_block(block: &Block) -> bool {
    block.namespace == "minecraft" && block.id == "lava"
}

fn is_intrinsic_fluid_block(block: &Block) -> bool {
    is_intrinsic_water_block(block) || is_intrinsic_lava_block(block)
}

fn raw_fluid_cell(block: &Block, fluid_kind: FluidKind) -> Option<FluidCell> {
    let fluid_id = match fluid_kind {
        FluidKind::Water => "water",
        FluidKind::Lava => "lava",
    };

    if block.namespace == "minecraft" && block.id == fluid_id {
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
        return Some(FluidCell {
            height: own_height,
            own_height,
            falling,
        });
    }

    if fluid_kind == FluidKind::Water
        && block.namespace == "minecraft"
        && block.id == "bubble_column"
    {
        return Some(FluidCell {
            height: 1.0,
            own_height: (8.0_f32 / 9.0_f32) as f64,
            falling: false,
        });
    }

    if fluid_kind == FluidKind::Water
        && block
            .attributes
            .get("waterlogged")
            .map(|value| value == "true")
            .unwrap_or(false)
    {
        return Some(FluidCell {
            height: 1.0,
            own_height: (8.0_f32 / 9.0_f32) as f64,
            falling: false,
        });
    }

    None
}

fn supported_partial_collision_kind(block: &Block) -> Option<CollisionKind> {
    if is_banner_block(block) {
        return Some(CollisionKind::NonSolid);
    }
    if block.id == "barrel" {
        return Some(CollisionKind::FullBlock);
    }
    if block.id.ends_with("candle") || is_amethyst_cluster_block(block) {
        return Some(CollisionKind::PartialBlock);
    }
    if is_simple_partial_environment_block(block) {
        return Some(CollisionKind::PartialBlock);
    }
    if matches!(
        block.id.as_str(),
        "brewing_stand"
            | "conduit"
            | "end_portal_frame"
            | "cocoa"
            | "stonecutter"
            | "cake"
            | "enchanting_table"
            | "daylight_detector"
            | "big_dripleaf"
            | "pointed_dripstone"
    ) {
        return Some(CollisionKind::PartialBlock);
    }
    if block.id.contains("cauldron")
        || block.id == "composter"
        || block.id == "lectern"
        || block.id == "bell"
        || block.id.contains("anvil")
    {
        return Some(CollisionKind::PartialBlock);
    }
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
    if block.id == "ladder"
        || block.id.ends_with("_chain")
        || block.id.ends_with("pressure_plate")
        || is_rod_block(block)
        || is_wall_hanging_sign_block(block)
        || is_ceiling_hanging_sign_block(block)
        || block.id.ends_with("trapdoor")
        || block.id.ends_with("door")
        || block.id.ends_with("stairs")
        || block.id.ends_with("pane")
        || block.id.ends_with("bars")
        || block.id.ends_with("wall")
        || block.id.ends_with("fence")
        || is_standing_sign_block(block)
        || is_wall_sign_block(block)
        || block.id.ends_with("carpet")
        || block.id == "snow"
        || block.id == "cactus"
        || block.id == "soul_sand"
        || block.id == "mud"
        || block.id == "honey_block"
        || block.id == "scaffolding"
        || block.id.ends_with("campfire")
        || block.id == "bamboo"
        || block.id == "decorated_pot"
        || block.id == "hopper"
        || is_chest_block(block)
        || is_standing_skull_or_head_block(block)
        || is_wall_skull_or_head_block(block)
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
    if is_banner_block(block) {
        return CollisionBoxes::empty();
    }
    if is_wall_hanging_sign_block(block) {
        return wall_hanging_sign_collision_boxes(block);
    }
    if is_ceiling_hanging_sign_block(block) {
        return ceiling_hanging_sign_collision_boxes(block);
    }
    if is_wall_sign_block(block) {
        return wall_sign_collision_boxes(block);
    }
    if is_standing_sign_block(block) {
        return standing_sign_collision_boxes();
    }
    if block.id == "ladder" {
        return ladder_collision_boxes(block);
    }
    if block.id == "bamboo" {
        return bamboo_collision_boxes();
    }
    if block.id.ends_with("_chain") {
        return chain_collision_boxes(block);
    }
    if is_rod_block(block) {
        return rod_collision_boxes(block);
    }
    if block.id.ends_with("candle") {
        return candle_collision_boxes(block);
    }
    if is_amethyst_cluster_block(block) {
        return amethyst_cluster_collision_boxes(block);
    }
    if block.id == "sea_pickle" {
        return sea_pickle_collision_boxes(block);
    }
    if block.id == "lily_pad" {
        return lily_pad_collision_boxes();
    }
    if block.id == "frogspawn" {
        return frogspawn_collision_boxes();
    }
    if block.id == "turtle_egg" {
        return turtle_egg_collision_boxes(block);
    }
    if is_flower_pot_block(block) {
        return flower_pot_collision_boxes();
    }
    if block.id == "brewing_stand" {
        return brewing_stand_collision_boxes();
    }
    if block.id == "conduit" {
        return conduit_collision_boxes();
    }
    if block.id == "end_portal_frame" {
        return end_portal_frame_collision_boxes(block);
    }
    if block.id == "cocoa" {
        return cocoa_collision_boxes(block);
    }
    if block.id == "stonecutter" {
        return stonecutter_collision_boxes();
    }
    if block.id == "cake" {
        return cake_collision_boxes(block);
    }
    if block.id == "enchanting_table" {
        return enchanting_table_collision_boxes();
    }
    if block.id == "daylight_detector" {
        return daylight_detector_collision_boxes();
    }
    if block.id == "big_dripleaf" {
        return big_dripleaf_collision_boxes(block);
    }
    if block.id == "pointed_dripstone" {
        return pointed_dripstone_collision_boxes(block);
    }
    if block.id.ends_with("pressure_plate") {
        return pressure_plate_collision_boxes(block);
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
    if block.id.ends_with("carpet") {
        return carpet_collision_boxes();
    }
    if block.id == "snow" {
        return snow_collision_boxes(block);
    }
    if block.id.ends_with("campfire") {
        return campfire_collision_boxes();
    }
    if block.id == "cactus" {
        return cactus_collision_boxes();
    }
    if block.id == "soul_sand" {
        return soul_sand_collision_boxes();
    }
    if block.id == "mud" {
        return soul_sand_collision_boxes();
    }
    if block.id == "honey_block" {
        return honey_block_collision_boxes();
    }
    if block.id == "scaffolding" {
        return scaffolding_collision_boxes();
    }
    if block.id.contains("cauldron") {
        return cauldron_collision_boxes();
    }
    if block.id == "composter" {
        return composter_collision_boxes();
    }
    if block.id == "lectern" {
        return lectern_collision_boxes();
    }
    if block.id == "bell" {
        return bell_collision_boxes(block);
    }
    if block.id.contains("anvil") {
        return anvil_collision_boxes(block);
    }
    if block.id == "decorated_pot" {
        return decorated_pot_collision_boxes();
    }
    if block.id == "hopper" {
        return hopper_collision_boxes(block);
    }
    if is_chest_block(block) {
        return chest_collision_boxes(block);
    }
    if is_standing_skull_or_head_block(block) {
        return standing_skull_collision_boxes(block);
    }
    if is_wall_skull_or_head_block(block) {
        return wall_skull_collision_boxes(block);
    }
    CollisionBoxes::empty()
}

fn is_rod_block(block: &Block) -> bool {
    block.namespace == "minecraft" && matches!(block.id.as_str(), "end_rod" | "lightning_rod")
}

fn is_standing_sign_block(block: &Block) -> bool {
    block.namespace == "minecraft"
        && block.id.ends_with("sign")
        && !block.id.ends_with("wall_sign")
        && !block.id.ends_with("hanging_sign")
}

fn is_wall_sign_block(block: &Block) -> bool {
    block.namespace == "minecraft" && block.id.ends_with("wall_sign")
}

fn is_wall_hanging_sign_block(block: &Block) -> bool {
    block.namespace == "minecraft" && block.id.ends_with("wall_hanging_sign")
}

fn is_ceiling_hanging_sign_block(block: &Block) -> bool {
    block.namespace == "minecraft"
        && block.id.ends_with("hanging_sign")
        && !block.id.ends_with("wall_hanging_sign")
}

fn is_banner_block(block: &Block) -> bool {
    block.namespace == "minecraft" && block.id.ends_with("banner")
}

fn is_chest_block(block: &Block) -> bool {
    block.namespace == "minecraft" && matches!(block.id.as_str(), "chest" | "trapped_chest")
}

fn is_standing_skull_or_head_block(block: &Block) -> bool {
    block.namespace == "minecraft"
        && (block.id.contains("skull") || block.id.contains("head"))
        && !block.id.contains("wall_")
}

fn is_wall_skull_or_head_block(block: &Block) -> bool {
    block.namespace == "minecraft"
        && (block.id.contains("skull") || block.id.contains("head"))
        && block.id.contains("wall_")
}

fn standing_sign_collision_boxes() -> CollisionBoxes {
    CollisionBoxes::single(centered_column_box(8.0 / 16.0, 1.0))
}

fn ladder_collision_boxes(block: &Block) -> CollisionBoxes {
    let facing = horizontal_direction(
        block
            .attributes
            .get("facing")
            .map(String::as_str)
            .unwrap_or("north"),
    );
    CollisionBoxes::single(rotate_y_clockwise(
        CollisionBox {
            min_x: 0.0,
            min_y: 0.0,
            min_z: 13.0 / 16.0,
            max_x: 1.0,
            max_y: 1.0,
            max_z: 1.0,
        },
        horizontal_turns(facing),
    ))
}

fn bamboo_collision_boxes() -> CollisionBoxes {
    CollisionBoxes::single(centered_column_box(3.0 / 16.0, 1.0))
}

fn chain_collision_boxes(block: &Block) -> CollisionBoxes {
    CollisionBoxes::single(axis_bar_box(block_axis_attr(block, "axis"), 3.0 / 16.0))
}

fn rod_collision_boxes(block: &Block) -> CollisionBoxes {
    CollisionBoxes::single(axis_bar_box(facing_axis(block), 4.0 / 16.0))
}

fn pressure_plate_collision_boxes(block: &Block) -> CollisionBoxes {
    let max_y = if pressure_plate_is_pressed(block) {
        0.5 / 16.0
    } else {
        1.0 / 16.0
    };
    CollisionBoxes::single(CollisionBox {
        min_x: 1.0 / 16.0,
        min_y: 0.0,
        min_z: 1.0 / 16.0,
        max_x: 15.0 / 16.0,
        max_y,
        max_z: 15.0 / 16.0,
    })
}

fn candle_collision_boxes(block: &Block) -> CollisionBoxes {
    let candles = block
        .attributes
        .get("candles")
        .and_then(|value| value.parse::<u8>().ok())
        .unwrap_or(1)
        .clamp(1, 4);
    let collision_box = match candles {
        1 => centered_column_segment_box(2.0 / 16.0, 0.0, 6.0 / 16.0),
        2 => CollisionBox {
            min_x: 5.0 / 16.0,
            min_y: 0.0,
            min_z: 6.0 / 16.0,
            max_x: 11.0 / 16.0,
            max_y: 6.0 / 16.0,
            max_z: 9.0 / 16.0,
        },
        3 => CollisionBox {
            min_x: 5.0 / 16.0,
            min_y: 0.0,
            min_z: 6.0 / 16.0,
            max_x: 10.0 / 16.0,
            max_y: 6.0 / 16.0,
            max_z: 11.0 / 16.0,
        },
        _ => CollisionBox {
            min_x: 5.0 / 16.0,
            min_y: 0.0,
            min_z: 5.0 / 16.0,
            max_x: 11.0 / 16.0,
            max_y: 6.0 / 16.0,
            max_z: 10.0 / 16.0,
        },
    };
    CollisionBoxes::single(collision_box)
}

fn sea_pickle_collision_boxes(block: &Block) -> CollisionBoxes {
    let pickles = block
        .attributes
        .get("pickles")
        .and_then(|value| value.parse::<u8>().ok())
        .unwrap_or(1)
        .clamp(1, 4);
    let (width, height) = match pickles {
        1 => (4.0 / 16.0, 6.0 / 16.0),
        2 => (10.0 / 16.0, 6.0 / 16.0),
        3 => (12.0 / 16.0, 6.0 / 16.0),
        _ => (12.0 / 16.0, 7.0 / 16.0),
    };
    CollisionBoxes::single(centered_column_segment_box(width, 0.0, height))
}

fn amethyst_cluster_collision_boxes(block: &Block) -> CollisionBoxes {
    let (width, height) = match block.id.as_str() {
        "small_amethyst_bud" => (8.0 / 16.0, 3.0 / 16.0),
        "medium_amethyst_bud" => (10.0 / 16.0, 4.0 / 16.0),
        "large_amethyst_bud" => (10.0 / 16.0, 5.0 / 16.0),
        _ => (10.0 / 16.0, 7.0 / 16.0),
    };
    let facing = block
        .attributes
        .get("facing")
        .map(String::as_str)
        .unwrap_or("up");
    let collision_box = match facing {
        "down" => centered_rect_segment_box(width, width, 1.0 - height, 1.0),
        "north" => CollisionBox {
            min_x: 0.5 - width * 0.5,
            min_y: 0.5 - width * 0.5,
            min_z: 1.0 - height,
            max_x: 0.5 + width * 0.5,
            max_y: 0.5 + width * 0.5,
            max_z: 1.0,
        },
        "south" => CollisionBox {
            min_x: 0.5 - width * 0.5,
            min_y: 0.5 - width * 0.5,
            min_z: 0.0,
            max_x: 0.5 + width * 0.5,
            max_y: 0.5 + width * 0.5,
            max_z: height,
        },
        "east" => CollisionBox {
            min_x: 0.0,
            min_y: 0.5 - width * 0.5,
            min_z: 0.5 - width * 0.5,
            max_x: height,
            max_y: 0.5 + width * 0.5,
            max_z: 0.5 + width * 0.5,
        },
        "west" => CollisionBox {
            min_x: 1.0 - height,
            min_y: 0.5 - width * 0.5,
            min_z: 0.5 - width * 0.5,
            max_x: 1.0,
            max_y: 0.5 + width * 0.5,
            max_z: 0.5 + width * 0.5,
        },
        _ => centered_rect_segment_box(width, width, 0.0, height),
    };
    CollisionBoxes::single(collision_box)
}

fn lily_pad_collision_boxes() -> CollisionBoxes {
    CollisionBoxes::single(centered_column_segment_box(14.0 / 16.0, 0.0, 1.5 / 16.0))
}

fn frogspawn_collision_boxes() -> CollisionBoxes {
    CollisionBoxes::single(CollisionBox {
        min_x: 0.0,
        min_y: 0.0,
        min_z: 0.0,
        max_x: 1.0,
        max_y: 1.5 / 16.0,
        max_z: 1.0,
    })
}

fn turtle_egg_collision_boxes(block: &Block) -> CollisionBoxes {
    let eggs = block
        .attributes
        .get("eggs")
        .and_then(|value| value.parse::<u8>().ok())
        .unwrap_or(1)
        .clamp(1, 4);
    if eggs <= 1 {
        return CollisionBoxes::single(CollisionBox {
            min_x: 3.0 / 16.0,
            min_y: 0.0,
            min_z: 3.0 / 16.0,
            max_x: 12.0 / 16.0,
            max_y: 7.0 / 16.0,
            max_z: 12.0 / 16.0,
        });
    }
    CollisionBoxes::single(centered_column_segment_box(14.0 / 16.0, 0.0, 7.0 / 16.0))
}

fn flower_pot_collision_boxes() -> CollisionBoxes {
    CollisionBoxes::single(centered_column_segment_box(6.0 / 16.0, 0.0, 6.0 / 16.0))
}

fn brewing_stand_collision_boxes() -> CollisionBoxes {
    let mut result = CollisionBoxes::empty();
    result.push(centered_column_segment_box(
        2.0 / 16.0,
        2.0 / 16.0,
        14.0 / 16.0,
    ));
    result.push(centered_column_segment_box(14.0 / 16.0, 0.0, 2.0 / 16.0));
    result
}

fn conduit_collision_boxes() -> CollisionBoxes {
    CollisionBoxes::single(centered_box(6.0 / 16.0, 6.0 / 16.0, 6.0 / 16.0))
}

fn end_portal_frame_collision_boxes(block: &Block) -> CollisionBoxes {
    let mut result = CollisionBoxes::empty();
    result.push(CollisionBox {
        min_x: 0.0,
        min_y: 0.0,
        min_z: 0.0,
        max_x: 1.0,
        max_y: 13.0 / 16.0,
        max_z: 1.0,
    });
    if bool_attr(block, "eye") {
        result.push(centered_column_segment_box(8.0 / 16.0, 13.0 / 16.0, 1.0));
    }
    result
}

fn cocoa_collision_boxes(block: &Block) -> CollisionBoxes {
    let age = block
        .attributes
        .get("age")
        .and_then(|value| value.parse::<u8>().ok())
        .unwrap_or(0)
        .clamp(0, 2);
    let facing = horizontal_direction(
        block
            .attributes
            .get("facing")
            .map(String::as_str)
            .unwrap_or("north"),
    );
    let width = (4.0 + age as f64 * 2.0) / 16.0;
    let min_y = (7.0 - age as f64 * 2.0) / 16.0;
    let max_y = 12.0 / 16.0;
    let z_shift = (age as f64 - 5.0) / 16.0;
    let north_box = CollisionBox {
        min_x: 0.5 - width * 0.5,
        min_y,
        min_z: 0.5 - width * 0.5 + z_shift,
        max_x: 0.5 + width * 0.5,
        max_y,
        max_z: 0.5 + width * 0.5 + z_shift,
    };
    CollisionBoxes::single(rotate_y_clockwise(north_box, horizontal_turns(facing)))
}

fn stonecutter_collision_boxes() -> CollisionBoxes {
    CollisionBoxes::single(CollisionBox {
        min_x: 0.0,
        min_y: 0.0,
        min_z: 0.0,
        max_x: 1.0,
        max_y: 9.0 / 16.0,
        max_z: 1.0,
    })
}

fn cake_collision_boxes(block: &Block) -> CollisionBoxes {
    let bites = block
        .attributes
        .get("bites")
        .and_then(|value| value.parse::<u8>().ok())
        .unwrap_or(0)
        .clamp(0, 6);
    CollisionBoxes::single(CollisionBox {
        min_x: (1.0 + bites as f64 * 2.0) / 16.0,
        min_y: 0.0,
        min_z: 1.0 / 16.0,
        max_x: 15.0 / 16.0,
        max_y: 8.0 / 16.0,
        max_z: 15.0 / 16.0,
    })
}

fn enchanting_table_collision_boxes() -> CollisionBoxes {
    CollisionBoxes::single(CollisionBox {
        min_x: 0.0,
        min_y: 0.0,
        min_z: 0.0,
        max_x: 1.0,
        max_y: 12.0 / 16.0,
        max_z: 1.0,
    })
}

fn daylight_detector_collision_boxes() -> CollisionBoxes {
    CollisionBoxes::single(CollisionBox {
        min_x: 0.0,
        min_y: 0.0,
        min_z: 0.0,
        max_x: 1.0,
        max_y: 6.0 / 16.0,
        max_z: 1.0,
    })
}

fn big_dripleaf_collision_boxes(block: &Block) -> CollisionBoxes {
    let max_y = match block
        .attributes
        .get("tilt")
        .map(String::as_str)
        .unwrap_or("none")
    {
        "full" => return CollisionBoxes::empty(),
        "partial" => 13.0 / 16.0,
        _ => 15.0 / 16.0,
    };
    CollisionBoxes::single(CollisionBox {
        min_x: 0.0,
        min_y: 11.0 / 16.0,
        min_z: 0.0,
        max_x: 1.0,
        max_y,
        max_z: 1.0,
    })
}

fn pointed_dripstone_collision_boxes(block: &Block) -> CollisionBoxes {
    let (width, min_y, max_y) = match block
        .attributes
        .get("thickness")
        .map(String::as_str)
        .unwrap_or("tip")
    {
        "tip_merge" => (6.0 / 16.0, 0.0, 1.0),
        "tip" => {
            if block
                .attributes
                .get("vertical_direction")
                .map(String::as_str)
                .unwrap_or("up")
                == "down"
            {
                (6.0 / 16.0, 5.0 / 16.0, 1.0)
            } else {
                (6.0 / 16.0, 0.0, 11.0 / 16.0)
            }
        }
        "frustum" => (8.0 / 16.0, 0.0, 1.0),
        "middle" => (10.0 / 16.0, 0.0, 1.0),
        "base" => (12.0 / 16.0, 0.0, 1.0),
        _ => (6.0 / 16.0, 0.0, 11.0 / 16.0),
    };
    CollisionBoxes::single(centered_rect_segment_box(width, width, min_y, max_y))
}

fn carpet_collision_boxes() -> CollisionBoxes {
    CollisionBoxes::single(CollisionBox {
        min_x: 0.0,
        min_y: 0.0,
        min_z: 0.0,
        max_x: 1.0,
        max_y: 1.0 / 16.0,
        max_z: 1.0,
    })
}

fn snow_collision_boxes(block: &Block) -> CollisionBoxes {
    let layers = block
        .attributes
        .get("layers")
        .and_then(|value| value.parse::<u8>().ok())
        .unwrap_or(1)
        .clamp(1, 8);
    if layers <= 1 {
        return CollisionBoxes::empty();
    }
    CollisionBoxes::single(CollisionBox {
        min_x: 0.0,
        min_y: 0.0,
        min_z: 0.0,
        max_x: 1.0,
        max_y: ((layers - 1) as f64) * (2.0 / 16.0),
        max_z: 1.0,
    })
}

fn snow_support_boxes(block: &Block) -> CollisionBoxes {
    let layers = block
        .attributes
        .get("layers")
        .and_then(|value| value.parse::<u8>().ok())
        .unwrap_or(1)
        .clamp(1, 8);
    CollisionBoxes::single(CollisionBox {
        min_x: 0.0,
        min_y: 0.0,
        min_z: 0.0,
        max_x: 1.0,
        max_y: (layers as f64) * (2.0 / 16.0),
        max_z: 1.0,
    })
}

fn cactus_collision_boxes() -> CollisionBoxes {
    CollisionBoxes::single(CollisionBox {
        min_x: 1.0 / 16.0,
        min_y: 0.0,
        min_z: 1.0 / 16.0,
        max_x: 15.0 / 16.0,
        max_y: 15.0 / 16.0,
        max_z: 15.0 / 16.0,
    })
}

fn campfire_collision_boxes() -> CollisionBoxes {
    CollisionBoxes::single(CollisionBox {
        min_x: 0.0,
        min_y: 0.0,
        min_z: 0.0,
        max_x: 1.0,
        max_y: 7.0 / 16.0,
        max_z: 1.0,
    })
}

fn soul_sand_collision_boxes() -> CollisionBoxes {
    CollisionBoxes::single(CollisionBox {
        min_x: 0.0,
        min_y: 0.0,
        min_z: 0.0,
        max_x: 1.0,
        max_y: 14.0 / 16.0,
        max_z: 1.0,
    })
}

fn honey_block_collision_boxes() -> CollisionBoxes {
    CollisionBoxes::single(CollisionBox {
        min_x: 1.0 / 16.0,
        min_y: 0.0,
        min_z: 1.0 / 16.0,
        max_x: 15.0 / 16.0,
        max_y: 15.0 / 16.0,
        max_z: 15.0 / 16.0,
    })
}

fn cauldron_collision_boxes() -> CollisionBoxes {
    let mut result = CollisionBoxes::empty();
    for (min_x, min_z) in [
        (0.0, 0.0),
        (14.0 / 16.0, 0.0),
        (0.0, 14.0 / 16.0),
        (14.0 / 16.0, 14.0 / 16.0),
    ] {
        result.push(CollisionBox {
            min_x,
            min_y: 0.0,
            min_z,
            max_x: min_x + 2.0 / 16.0,
            max_y: 3.0 / 16.0,
            max_z: min_z + 2.0 / 16.0,
        });
    }
    result.push(CollisionBox {
        min_x: 0.0,
        min_y: 3.0 / 16.0,
        min_z: 0.0,
        max_x: 1.0,
        max_y: 1.0,
        max_z: 2.0 / 16.0,
    });
    result.push(CollisionBox {
        min_x: 0.0,
        min_y: 3.0 / 16.0,
        min_z: 14.0 / 16.0,
        max_x: 1.0,
        max_y: 1.0,
        max_z: 1.0,
    });
    result.push(CollisionBox {
        min_x: 0.0,
        min_y: 3.0 / 16.0,
        min_z: 2.0 / 16.0,
        max_x: 2.0 / 16.0,
        max_y: 1.0,
        max_z: 14.0 / 16.0,
    });
    result.push(CollisionBox {
        min_x: 14.0 / 16.0,
        min_y: 3.0 / 16.0,
        min_z: 2.0 / 16.0,
        max_x: 1.0,
        max_y: 1.0,
        max_z: 14.0 / 16.0,
    });
    result
}

fn composter_collision_boxes() -> CollisionBoxes {
    let mut result = CollisionBoxes::empty();
    result.push(CollisionBox {
        min_x: 0.0,
        min_y: 0.0,
        min_z: 0.0,
        max_x: 1.0,
        max_y: 2.0 / 16.0,
        max_z: 1.0,
    });
    result.push(CollisionBox {
        min_x: 0.0,
        min_y: 2.0 / 16.0,
        min_z: 0.0,
        max_x: 1.0,
        max_y: 1.0,
        max_z: 2.0 / 16.0,
    });
    result.push(CollisionBox {
        min_x: 0.0,
        min_y: 2.0 / 16.0,
        min_z: 14.0 / 16.0,
        max_x: 1.0,
        max_y: 1.0,
        max_z: 1.0,
    });
    result.push(CollisionBox {
        min_x: 0.0,
        min_y: 2.0 / 16.0,
        min_z: 2.0 / 16.0,
        max_x: 2.0 / 16.0,
        max_y: 1.0,
        max_z: 14.0 / 16.0,
    });
    result.push(CollisionBox {
        min_x: 14.0 / 16.0,
        min_y: 2.0 / 16.0,
        min_z: 2.0 / 16.0,
        max_x: 1.0,
        max_y: 1.0,
        max_z: 14.0 / 16.0,
    });
    result
}

fn lectern_collision_boxes() -> CollisionBoxes {
    let mut result = CollisionBoxes::empty();
    result.push(CollisionBox {
        min_x: 0.0,
        min_y: 0.0,
        min_z: 0.0,
        max_x: 1.0,
        max_y: 2.0 / 16.0,
        max_z: 1.0,
    });
    result.push(centered_column_segment_box(
        8.0 / 16.0,
        2.0 / 16.0,
        14.0 / 16.0,
    ));
    result
}

fn bell_collision_boxes(block: &Block) -> CollisionBoxes {
    let mut result = bell_body_collision_boxes();
    let facing = horizontal_direction(
        block
            .attributes
            .get("facing")
            .map(String::as_str)
            .unwrap_or("north"),
    );
    match block
        .attributes
        .get("attachment")
        .map(String::as_str)
        .unwrap_or("floor")
    {
        "ceiling" => {
            result.push(centered_column_segment_box(2.0 / 16.0, 13.0 / 16.0, 1.0));
        }
        "double_wall" => {
            let box_shape = if horizontal_axis_is_x(facing) {
                centered_rect_segment_box(1.0, 2.0 / 16.0, 13.0 / 16.0, 15.0 / 16.0)
            } else {
                centered_rect_segment_box(2.0 / 16.0, 1.0, 13.0 / 16.0, 15.0 / 16.0)
            };
            result.push(box_shape);
        }
        "single_wall" => {
            result.push(rotate_y_clockwise(
                CollisionBox {
                    min_x: 7.0 / 16.0,
                    min_y: 13.0 / 16.0,
                    min_z: 0.0,
                    max_x: 9.0 / 16.0,
                    max_y: 15.0 / 16.0,
                    max_z: 13.0 / 16.0,
                },
                horizontal_turns(facing),
            ));
        }
        _ => {
            let box_shape = if horizontal_axis_is_x(facing) {
                centered_rect_segment_box(1.0, 8.0 / 16.0, 0.0, 1.0)
            } else {
                centered_rect_segment_box(8.0 / 16.0, 1.0, 0.0, 1.0)
            };
            result.push(box_shape);
        }
    }
    result
}

fn bell_body_collision_boxes() -> CollisionBoxes {
    let mut result = CollisionBoxes::empty();
    result.push(centered_column_segment_box(
        6.0 / 16.0,
        6.0 / 16.0,
        13.0 / 16.0,
    ));
    result.push(centered_column_segment_box(
        8.0 / 16.0,
        4.0 / 16.0,
        6.0 / 16.0,
    ));
    result
}

fn anvil_collision_boxes(block: &Block) -> CollisionBoxes {
    let rotate = match block
        .attributes
        .get("facing")
        .map(String::as_str)
        .unwrap_or("north")
    {
        "east" | "west" => 1,
        _ => 0,
    };
    let mut result = CollisionBoxes::empty();
    for collision_box in [
        centered_rect_segment_box(12.0 / 16.0, 12.0 / 16.0, 0.0, 4.0 / 16.0),
        centered_rect_segment_box(8.0 / 16.0, 10.0 / 16.0, 4.0 / 16.0, 5.0 / 16.0),
        centered_rect_segment_box(4.0 / 16.0, 8.0 / 16.0, 5.0 / 16.0, 10.0 / 16.0),
        centered_rect_segment_box(10.0 / 16.0, 1.0, 10.0 / 16.0, 1.0),
    ] {
        result.push(rotate_y_clockwise(collision_box, rotate));
    }
    result
}

fn scaffolding_collision_boxes() -> CollisionBoxes {
    let mut result = CollisionBoxes::empty();
    result.push(CollisionBox {
        min_x: 0.0,
        min_y: 14.0 / 16.0,
        min_z: 0.0,
        max_x: 1.0,
        max_y: 1.0,
        max_z: 1.0,
    });
    for (min_x, min_z) in [
        (0.0, 0.0),
        (14.0 / 16.0, 0.0),
        (0.0, 14.0 / 16.0),
        (14.0 / 16.0, 14.0 / 16.0),
    ] {
        result.push(CollisionBox {
            min_x,
            min_y: 0.0,
            min_z,
            max_x: min_x + 2.0 / 16.0,
            max_y: 1.0,
            max_z: min_z + 2.0 / 16.0,
        });
    }
    result
}

fn decorated_pot_collision_boxes() -> CollisionBoxes {
    CollisionBoxes::single(CollisionBox {
        min_x: 1.0 / 16.0,
        min_y: 0.0,
        min_z: 1.0 / 16.0,
        max_x: 15.0 / 16.0,
        max_y: 1.0,
        max_z: 15.0 / 16.0,
    })
}

fn hopper_collision_boxes(block: &Block) -> CollisionBoxes {
    let mut result = CollisionBoxes::empty();
    result.push(CollisionBox {
        min_x: 0.0,
        min_y: 10.0 / 16.0,
        min_z: 0.0,
        max_x: 1.0,
        max_y: 11.0 / 16.0,
        max_z: 1.0,
    });
    result.push(CollisionBox {
        min_x: 0.0,
        min_y: 11.0 / 16.0,
        min_z: 0.0,
        max_x: 1.0,
        max_y: 1.0,
        max_z: 2.0 / 16.0,
    });
    result.push(CollisionBox {
        min_x: 0.0,
        min_y: 11.0 / 16.0,
        min_z: 14.0 / 16.0,
        max_x: 1.0,
        max_y: 1.0,
        max_z: 1.0,
    });
    result.push(CollisionBox {
        min_x: 0.0,
        min_y: 11.0 / 16.0,
        min_z: 2.0 / 16.0,
        max_x: 2.0 / 16.0,
        max_y: 1.0,
        max_z: 14.0 / 16.0,
    });
    result.push(CollisionBox {
        min_x: 14.0 / 16.0,
        min_y: 11.0 / 16.0,
        min_z: 2.0 / 16.0,
        max_x: 1.0,
        max_y: 1.0,
        max_z: 14.0 / 16.0,
    });
    result.push(CollisionBox {
        min_x: 4.0 / 16.0,
        min_y: 4.0 / 16.0,
        min_z: 4.0 / 16.0,
        max_x: 12.0 / 16.0,
        max_y: 10.0 / 16.0,
        max_z: 12.0 / 16.0,
    });
    result.push(
        match block
            .attributes
            .get("facing")
            .map(String::as_str)
            .unwrap_or("down")
        {
            "north" => CollisionBox {
                min_x: 6.0 / 16.0,
                min_y: 4.0 / 16.0,
                min_z: 0.0,
                max_x: 10.0 / 16.0,
                max_y: 8.0 / 16.0,
                max_z: 8.0 / 16.0,
            },
            "east" => CollisionBox {
                min_x: 8.0 / 16.0,
                min_y: 4.0 / 16.0,
                min_z: 6.0 / 16.0,
                max_x: 1.0,
                max_y: 8.0 / 16.0,
                max_z: 10.0 / 16.0,
            },
            "south" => CollisionBox {
                min_x: 6.0 / 16.0,
                min_y: 4.0 / 16.0,
                min_z: 8.0 / 16.0,
                max_x: 10.0 / 16.0,
                max_y: 8.0 / 16.0,
                max_z: 1.0,
            },
            "west" => CollisionBox {
                min_x: 0.0,
                min_y: 4.0 / 16.0,
                min_z: 6.0 / 16.0,
                max_x: 8.0 / 16.0,
                max_y: 8.0 / 16.0,
                max_z: 10.0 / 16.0,
            },
            _ => CollisionBox {
                min_x: 6.0 / 16.0,
                min_y: 0.0,
                min_z: 6.0 / 16.0,
                max_x: 10.0 / 16.0,
                max_y: 8.0 / 16.0,
                max_z: 10.0 / 16.0,
            },
        },
    );
    result
}

fn chest_collision_boxes(block: &Block) -> CollisionBoxes {
    let chest_type = block
        .attributes
        .get("type")
        .map(String::as_str)
        .unwrap_or("single");
    if chest_type == "single" {
        return CollisionBoxes::single(CollisionBox {
            min_x: 1.0 / 16.0,
            min_y: 0.0,
            min_z: 1.0 / 16.0,
            max_x: 15.0 / 16.0,
            max_y: 14.0 / 16.0,
            max_z: 15.0 / 16.0,
        });
    }

    let facing = horizontal_direction(
        block
            .attributes
            .get("facing")
            .map(String::as_str)
            .unwrap_or("north"),
    );
    let connected = if chest_type == "left" {
        rotate_clockwise(facing)
    } else {
        rotate_counter_clockwise(facing)
    };
    CollisionBoxes::single(rotate_y_clockwise(
        CollisionBox {
            min_x: 1.0 / 16.0,
            min_y: 0.0,
            min_z: 0.0,
            max_x: 15.0 / 16.0,
            max_y: 14.0 / 16.0,
            max_z: 15.0 / 16.0,
        },
        horizontal_turns(connected),
    ))
}

fn standing_skull_collision_boxes(block: &Block) -> CollisionBoxes {
    let width = if block.id.contains("piglin") {
        10.0 / 16.0
    } else {
        8.0 / 16.0
    };
    CollisionBoxes::single(centered_column_box(width, 8.0 / 16.0))
}

fn wall_skull_collision_boxes(block: &Block) -> CollisionBoxes {
    let facing = horizontal_direction(
        block
            .attributes
            .get("facing")
            .map(String::as_str)
            .unwrap_or("north"),
    );
    CollisionBoxes::single(rotate_y_clockwise(
        CollisionBox {
            min_x: 4.0 / 16.0,
            min_y: 4.0 / 16.0,
            min_z: 8.0 / 16.0,
            max_x: 12.0 / 16.0,
            max_y: 12.0 / 16.0,
            max_z: 1.0,
        },
        horizontal_turns(facing),
    ))
}

fn wall_hanging_sign_collision_boxes(block: &Block) -> CollisionBoxes {
    CollisionBoxes::single(match facing_axis(block) {
        BlockAxis::X => CollisionBox {
            min_x: 6.0 / 16.0,
            min_y: 14.0 / 16.0,
            min_z: 0.0,
            max_x: 10.0 / 16.0,
            max_y: 1.0,
            max_z: 1.0,
        },
        BlockAxis::Y => CollisionBox::default(),
        BlockAxis::Z => CollisionBox {
            min_x: 0.0,
            min_y: 14.0 / 16.0,
            min_z: 6.0 / 16.0,
            max_x: 1.0,
            max_y: 1.0,
            max_z: 10.0 / 16.0,
        },
    })
}

fn ceiling_hanging_sign_collision_boxes(block: &Block) -> CollisionBoxes {
    let rotation = block
        .attributes
        .get("rotation")
        .and_then(|value| value.parse::<u8>().ok())
        .unwrap_or(0);
    let collision_box = match rotation {
        0 | 8 => CollisionBox {
            min_x: 1.0 / 16.0,
            min_y: 0.0,
            min_z: 7.0 / 16.0,
            max_x: 15.0 / 16.0,
            max_y: 10.0 / 16.0,
            max_z: 9.0 / 16.0,
        },
        4 | 12 => CollisionBox {
            min_x: 7.0 / 16.0,
            min_y: 0.0,
            min_z: 1.0 / 16.0,
            max_x: 9.0 / 16.0,
            max_y: 10.0 / 16.0,
            max_z: 15.0 / 16.0,
        },
        _ => centered_box(10.0 / 16.0, 1.0, 10.0 / 16.0),
    };
    CollisionBoxes::single(collision_box)
}

fn wall_sign_collision_boxes(block: &Block) -> CollisionBoxes {
    let facing = horizontal_direction(
        block
            .attributes
            .get("facing")
            .map(String::as_str)
            .unwrap_or("north"),
    );
    CollisionBoxes::single(rotate_y_clockwise(
        CollisionBox {
            min_x: 0.0,
            min_y: 4.5 / 16.0,
            min_z: 14.0 / 16.0,
            max_x: 1.0,
            max_y: 12.5 / 16.0,
            max_z: 1.0,
        },
        horizontal_turns(facing),
    ))
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

fn centered_column_segment_box(width: f64, min_y: f64, max_y: f64) -> CollisionBox {
    CollisionBox {
        min_x: 0.5 - width * 0.5,
        min_y,
        min_z: 0.5 - width * 0.5,
        max_x: 0.5 + width * 0.5,
        max_y,
        max_z: 0.5 + width * 0.5,
    }
}

fn centered_rect_segment_box(width_x: f64, width_z: f64, min_y: f64, max_y: f64) -> CollisionBox {
    CollisionBox {
        min_x: 0.5 - width_x * 0.5,
        min_y,
        min_z: 0.5 - width_z * 0.5,
        max_x: 0.5 + width_x * 0.5,
        max_y,
        max_z: 0.5 + width_z * 0.5,
    }
}

fn centered_box(width_x: f64, height: f64, width_z: f64) -> CollisionBox {
    CollisionBox {
        min_x: 0.5 - width_x * 0.5,
        min_y: 0.5 - height * 0.5,
        min_z: 0.5 - width_z * 0.5,
        max_x: 0.5 + width_x * 0.5,
        max_y: 0.5 + height * 0.5,
        max_z: 0.5 + width_z * 0.5,
    }
}

#[derive(Clone, Copy)]
enum BlockAxis {
    X,
    Y,
    Z,
}

fn axis_bar_box(axis: BlockAxis, thickness: f64) -> CollisionBox {
    match axis {
        BlockAxis::X => centered_box(1.0, thickness, thickness),
        BlockAxis::Y => centered_box(thickness, 1.0, thickness),
        BlockAxis::Z => centered_box(thickness, thickness, 1.0),
    }
}

fn block_axis_attr(block: &Block, key: &str) -> BlockAxis {
    match block.attributes.get(key).map(String::as_str).unwrap_or("y") {
        "x" => BlockAxis::X,
        "z" => BlockAxis::Z,
        _ => BlockAxis::Y,
    }
}

fn facing_axis(block: &Block) -> BlockAxis {
    match block
        .attributes
        .get("facing")
        .map(String::as_str)
        .unwrap_or("north")
    {
        "east" | "west" => BlockAxis::X,
        "up" | "down" => BlockAxis::Y,
        _ => BlockAxis::Z,
    }
}

fn pressure_plate_is_pressed(block: &Block) -> bool {
    bool_attr(block, "powered")
        || block
            .attributes
            .get("power")
            .and_then(|value| value.parse::<u8>().ok())
            .map(|value| value > 0)
            .unwrap_or(false)
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
            | "sweet_berry_bush"
            | "tripwire"
            | "redstone_wire"
            | "nether_portal"
            | "end_gateway"
            | "end_portal"
            | "structure_void"
            | "light"
            | "fire"
            | "soul_fire"
            | "cobweb"
            | "powder_snow"
            | "lever"
            | "torch"
            | "wall_torch"
            | "redstone_torch"
            | "redstone_wall_torch"
    ) || id.ends_with("sign")
        || id.contains("pressure_plate")
        || id.contains("button")
        || id.contains("rail")
        || is_coral_non_solid_block_id(id)
        || is_no_collision_plant_block_id(id)
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
        || id.ends_with("candle")
        || is_amethyst_cluster_block_id(id)
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
                | "big_dripleaf"
                | "pointed_dripstone"
                | "bamboo"
        )
}

fn is_coral_non_solid_block_id(id: &str) -> bool {
    id.ends_with("_coral") || id.ends_with("_coral_fan") || id.ends_with("_coral_wall_fan")
}

fn is_simple_partial_environment_block(block: &Block) -> bool {
    block.namespace == "minecraft"
        && matches!(
            block.id.as_str(),
            "sea_pickle" | "lily_pad" | "frogspawn" | "turtle_egg"
        )
        || is_flower_pot_block(block)
}

fn is_flower_pot_block(block: &Block) -> bool {
    block.namespace == "minecraft" && (block.id == "flower_pot" || block.id.starts_with("potted_"))
}

fn is_no_collision_plant_block_id(id: &str) -> bool {
    id.ends_with("_sapling")
        || matches!(
            id,
            "mangrove_propagule"
                | "poppy"
                | "dandelion"
                | "blue_orchid"
                | "allium"
                | "azure_bluet"
                | "oxeye_daisy"
                | "cornflower"
                | "lily_of_the_valley"
                | "wither_rose"
                | "torchflower"
                | "sunflower"
                | "lilac"
                | "rose_bush"
                | "peony"
                | "wheat"
                | "carrots"
                | "potatoes"
                | "beetroots"
                | "nether_wart"
                | "pumpkin_stem"
                | "melon_stem"
                | "attached_pumpkin_stem"
                | "attached_melon_stem"
                | "cave_vines"
                | "cave_vines_plant"
                | "spore_blossom"
                | "pink_petals"
                | "wildflowers"
                | "leaf_litter"
                | "small_dripleaf"
                | "big_dripleaf_stem"
                | "hanging_roots"
                | "crimson_roots"
                | "warped_roots"
                | "nether_sprouts"
        )
        || id.ends_with("_tulip")
}

fn is_amethyst_cluster_block(block: &Block) -> bool {
    is_amethyst_cluster_block_id(block.id.as_str())
}

fn is_amethyst_cluster_block_id(id: &str) -> bool {
    matches!(
        id,
        "amethyst_cluster" | "small_amethyst_bud" | "medium_amethyst_bud" | "large_amethyst_bud"
    )
}
