use crate::litematic::{
    CollisionBox, FaceDirection, LoadedSchematic, WaterCell, block_at, block_full_id,
    blocks_motion, collision_boxes, is_collision_shape_full_block, is_face_sturdy, load_litematic,
    merged_face_occludes, water_at,
};
use crate::{
    AABB_DEFLATE, BUOYANCY, BUOYANCY_CAP, FLUID_CURRENT_EPSILON2, FLUID_CURRENT_MIN_IMPULSE,
    FLUID_CURRENT_MIN_OLD_MOVEMENT, FLUID_MOVEMENT_THRESHOLD, GRAVITY, HORIZONTAL_MOVEMENT_DAMPING,
    HORIZONTAL_REST_THRESHOLD2, HORIZONTAL_WATER_DAMPING, MOVEMENT_SAMPLE_MODULO,
    SLIME_STEP_ON_BASE, SLIME_STEP_ON_VY_SCALE, SLIME_STEP_ON_VY_THRESHOLD, VERIFY_DEFAULT_HEIGHT,
    VERIFY_DEFAULT_WIDTH, VERTICAL_MOVEMENT_DAMPING, VerifyCommand, WATER_PUSH,
};
use mc_schem::region::WorldSlice;
use mc_schem::{Block, Region};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;
use time::OffsetDateTime;
use time::format_description::well_known::Rfc3339;

const VERIFY_COLLISION_MODEL: &str = "axis_aligned_block_shapes_with_supported_partials";
const VERIFY_FLUID_MODEL: &str = "guardian_base_tick_plus_post_move_water_current";
const NO_PHYSICS_DEFLATE: f64 = 1.0e-7;
const NO_PHYSICS_PUSHOUT_SPEED: f64 = 0.2;
const FLOWING_WATER_TICK_DELAY: usize = 5;
const WATER_SLOPE_FIND_DISTANCE: usize = 4;

#[derive(Clone, Copy, Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
struct Vec3d {
    x: f64,
    y: f64,
    z: f64,
}

impl Vec3d {
    const ZERO: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    fn add(self, other: Self) -> Self {
        Self::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }

    fn scale(self, factor: f64) -> Self {
        Self::new(self.x * factor, self.y * factor, self.z * factor)
    }

    fn length_sqr(self) -> f64 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    fn length(self) -> f64 {
        self.length_sqr().sqrt()
    }

    fn horizontal_length_sqr(self) -> f64 {
        self.x * self.x + self.z * self.z
    }

    fn horizontal_length(self) -> f64 {
        self.horizontal_length_sqr().sqrt()
    }

    fn normalized(self) -> Self {
        let length = self.length();
        if length <= 1.0e-12 {
            Self::ZERO
        } else {
            self.scale(1.0 / length)
        }
    }

    fn dot_horizontal(self, other: Self) -> f64 {
        self.x * other.x + self.z * other.z
    }
}

#[derive(Clone, Copy, Debug)]
struct GroundProfile {
    friction: f64,
    slime: bool,
}

#[derive(Clone, Copy, Debug, Default)]
struct WaterTracker {
    height: f64,
    accumulated_current: Vec3d,
    current_count: usize,
}

impl WaterTracker {
    fn is_in_water(self) -> bool {
        self.height > 0.0
    }

    fn applies_underwater_movement(self) -> bool {
        self.height > FLUID_MOVEMENT_THRESHOLD
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DynamicWaterState {
    amount: u8,
    falling: bool,
}

impl DynamicWaterState {
    fn from_cell(cell: WaterCell) -> Self {
        Self {
            amount: (cell.own_height * 9.0).round().clamp(0.0, 8.0) as u8,
            falling: cell.falling,
        }
    }

    fn is_source(self) -> bool {
        self.amount == 8 && !self.falling
    }

    fn legacy_level(self) -> u8 {
        if self.is_source() {
            0
        } else {
            8_u8.saturating_sub(self.amount.min(8)) + if self.falling { 8 } else { 0 }
        }
    }

    fn to_block(self) -> Block {
        if self.is_source() {
            Block::from_id("minecraft:water").expect("valid water source")
        } else {
            Block::from_id(&format!("minecraft:water[level={}]", self.legacy_level()))
                .expect("valid flowing water")
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HorizontalDir {
    North,
    South,
    West,
    East,
}

impl HorizontalDir {
    const ALL: [Self; 4] = [Self::North, Self::South, Self::West, Self::East];

    fn offset(self) -> [i32; 3] {
        match self {
            Self::North => [0, 0, -1],
            Self::South => [0, 0, 1],
            Self::West => [-1, 0, 0],
            Self::East => [1, 0, 0],
        }
    }

    fn face(self) -> FaceDirection {
        match self {
            Self::North => FaceDirection::North,
            Self::South => FaceDirection::South,
            Self::West => FaceDirection::West,
            Self::East => FaceDirection::East,
        }
    }

    fn opposite(self) -> Self {
        match self {
            Self::North => Self::South,
            Self::South => Self::North,
            Self::West => Self::East,
            Self::East => Self::West,
        }
    }
}

#[derive(Default)]
struct DynamicFluidTicks {
    scheduled: BTreeMap<usize, BTreeSet<[i32; 3]>>,
}

impl DynamicFluidTicks {
    fn bootstrap(region: &Region) -> Self {
        let mut ticks = Self::default();
        let shape = region.shape();
        for x in 0..shape[0] {
            for y in 0..shape[1] {
                for z in 0..shape[2] {
                    let pos = [x, y, z];
                    if let Some(fluid) = dynamic_water_state_at(region, pos) {
                        let should_schedule = !fluid.is_source()
                            || [
                                [0, -1, 0],
                                [0, 1, 0],
                                [-1, 0, 0],
                                [1, 0, 0],
                                [0, 0, -1],
                                [0, 0, 1],
                            ]
                            .into_iter()
                            .any(|offset| {
                                dynamic_water_state_at(region, offset_pos(pos, offset))
                                    .map(|neighbor| !neighbor.is_source())
                                    .unwrap_or(false)
                            });
                        if should_schedule {
                            ticks.schedule(FLOWING_WATER_TICK_DELAY, pos);
                        }
                    }
                }
            }
        }
        ticks
    }

    fn schedule(&mut self, tick: usize, pos: [i32; 3]) {
        self.scheduled.entry(tick).or_default().insert(pos);
    }

    fn schedule_neighbors(&mut self, tick: usize, pos: [i32; 3]) {
        self.schedule(tick, pos);
        for offset in [
            [0, -1, 0],
            [0, 1, 0],
            [-1, 0, 0],
            [1, 0, 0],
            [0, 0, -1],
            [0, 0, 1],
        ] {
            self.schedule(tick, offset_pos(pos, offset));
        }
    }

    fn run_due(&mut self, region: &mut Region, tick: usize) {
        let Some(due) = self.scheduled.remove(&tick) else {
            return;
        };
        for pos in due {
            self.tick_water(region, pos, tick);
        }
    }

    fn tick_water(&mut self, region: &mut Region, pos: [i32; 3], tick: usize) {
        let Some(mut fluid) = dynamic_water_state_at(region, pos) else {
            return;
        };
        let Some(mut block_state) = block_at(region, pos).cloned() else {
            return;
        };

        if !fluid.is_source() {
            match get_new_water(region, pos, &block_state) {
                Some(new_fluid) if new_fluid != fluid => {
                    fluid = new_fluid;
                    set_water_block(region, pos, new_fluid);
                    self.schedule(tick + FLOWING_WATER_TICK_DELAY, pos);
                    block_state = block_at(region, pos).cloned().expect("water block exists");
                }
                None => {
                    set_air_block(region, pos);
                    self.schedule_neighbors(tick + FLOWING_WATER_TICK_DELAY, pos);
                    return;
                }
                _ => {}
            }
        }

        self.spread(region, pos, &block_state, fluid, tick);
    }

    fn spread(
        &mut self,
        region: &mut Region,
        pos: [i32; 3],
        block_state: &Block,
        fluid: DynamicWaterState,
        tick: usize,
    ) {
        let below_pos = offset_pos(pos, [0, -1, 0]);
        if let Some(below_block) = block_at(region, below_pos).cloned() {
            let below_fluid = dynamic_water_state_at(region, below_pos);
            if can_maybe_pass_through(
                region,
                pos,
                block_state,
                below_pos,
                &below_block,
                below_fluid,
                None,
            ) {
                if let Some(new_below_fluid) = get_new_water(region, below_pos, &below_block) {
                    if below_fluid.is_none() && can_hold_specific_fluid(&below_block) {
                        self.spread_to(region, below_pos, &below_block, new_below_fluid, tick);
                        if source_neighbor_count(region, pos) >= 3 {
                            self.spread_to_sides(region, pos, fluid, block_state, tick);
                        }
                        return;
                    }
                }
            }

            if fluid.is_source()
                || !is_water_hole(region, pos, block_state, below_pos, &below_block)
            {
                self.spread_to_sides(region, pos, fluid, block_state, tick);
            }
        }
    }

    fn spread_to_sides(
        &mut self,
        region: &mut Region,
        pos: [i32; 3],
        fluid: DynamicWaterState,
        block_state: &Block,
        tick: usize,
    ) {
        let mut neighbor_amount = fluid.amount.saturating_sub(1);
        if fluid.falling {
            neighbor_amount = 7;
        }
        if neighbor_amount == 0 {
            return;
        }

        for (direction, new_fluid) in get_spread(region, pos, block_state) {
            let neighbor_pos = offset_pos(pos, direction.offset());
            if let Some(neighbor_block) = block_at(region, neighbor_pos).cloned() {
                self.spread_to(region, neighbor_pos, &neighbor_block, new_fluid, tick);
            }
        }
    }

    fn spread_to(
        &mut self,
        region: &mut Region,
        pos: [i32; 3],
        state: &Block,
        fluid: DynamicWaterState,
        tick: usize,
    ) {
        if !try_place_waterlogged_block(region, pos, state) {
            set_water_block(region, pos, fluid);
        }
        self.schedule_neighbors(tick + FLOWING_WATER_TICK_DELAY, pos);
    }
}

fn dynamic_water_state_at(region: &Region, pos: [i32; 3]) -> Option<DynamicWaterState> {
    water_at(region, pos).map(DynamicWaterState::from_cell)
}

fn set_water_block(region: &mut Region, pos: [i32; 3], fluid: DynamicWaterState) {
    let block = fluid.to_block();
    let _ = region.set_block(pos, &block);
}

fn set_air_block(region: &mut Region, pos: [i32; 3]) {
    let air = Block::from_id("minecraft:air").expect("valid air block");
    let _ = region.set_block(pos, &air);
}

fn offset_pos(pos: [i32; 3], offset: [i32; 3]) -> [i32; 3] {
    [pos[0] + offset[0], pos[1] + offset[1], pos[2] + offset[2]]
}

fn get_new_water(region: &Region, pos: [i32; 3], state: &Block) -> Option<DynamicWaterState> {
    let mut highest_neighbor = 0_u8;
    let mut neighbour_sources = 0;

    for direction in HorizontalDir::ALL {
        let relative_pos = offset_pos(pos, direction.offset());
        let Some(block_state) = block_at(region, relative_pos) else {
            continue;
        };
        let Some(fluid_state) = dynamic_water_state_at(region, relative_pos) else {
            continue;
        };
        if can_pass_through_wall(region, pos, state, relative_pos, block_state, direction)
            && fluid_state.amount > 0
        {
            if fluid_state.is_source() {
                neighbour_sources += 1;
            }
            highest_neighbor = highest_neighbor.max(fluid_state.amount);
        }
    }

    if neighbour_sources >= 2 {
        let below_pos = offset_pos(pos, [0, -1, 0]);
        if let Some(below_block) = block_at(region, below_pos) {
            let below_fluid = dynamic_water_state_at(region, below_pos);
            if blocks_motion(below_block)
                || below_fluid.map(|fluid| fluid.is_source()).unwrap_or(false)
            {
                return Some(DynamicWaterState {
                    amount: 8,
                    falling: false,
                });
            }
        }
    }

    let above_pos = offset_pos(pos, [0, 1, 0]);
    if let Some(above_block) = block_at(region, above_pos) {
        if let Some(above_fluid) = dynamic_water_state_at(region, above_pos) {
            if can_pass_through_wall_vertical(
                region,
                pos,
                state,
                above_pos,
                above_block,
                FaceDirection::Up,
            ) && above_fluid.amount > 0
            {
                return Some(DynamicWaterState {
                    amount: 8,
                    falling: true,
                });
            }
        }
    }

    let amount = highest_neighbor.saturating_sub(1);
    if amount == 0 {
        None
    } else {
        Some(DynamicWaterState {
            amount,
            falling: false,
        })
    }
}

fn source_neighbor_count(region: &Region, pos: [i32; 3]) -> usize {
    HorizontalDir::ALL
        .into_iter()
        .filter(|direction| {
            dynamic_water_state_at(region, offset_pos(pos, direction.offset()))
                .map(|fluid| fluid.is_source())
                .unwrap_or(false)
        })
        .count()
}

fn get_spread(
    region: &Region,
    pos: [i32; 3],
    state: &Block,
) -> Vec<(HorizontalDir, DynamicWaterState)> {
    let mut lowest = usize::MAX;
    let mut result = Vec::new();

    for direction in HorizontalDir::ALL {
        let test_pos = offset_pos(pos, direction.offset());
        let Some(test_state) = block_at(region, test_pos) else {
            continue;
        };
        let test_fluid = dynamic_water_state_at(region, test_pos);
        if !can_maybe_pass_through(
            region,
            pos,
            state,
            test_pos,
            test_state,
            test_fluid,
            Some(direction),
        ) {
            continue;
        }
        let Some(new_fluid) = get_new_water(region, test_pos, test_state) else {
            continue;
        };
        if !can_hold_specific_fluid(test_state) || test_fluid.is_some() {
            continue;
        }

        let distance = if is_hole(region, test_pos, test_state) {
            0
        } else {
            get_slope_distance(region, test_pos, 1, direction.opposite(), test_state)
        };
        if distance < lowest {
            result.clear();
            lowest = distance;
        }
        if distance <= lowest {
            result.push((direction, new_fluid));
        }
    }

    result
}

fn get_slope_distance(
    region: &Region,
    pos: [i32; 3],
    pass: usize,
    from: HorizontalDir,
    state: &Block,
) -> usize {
    let mut lowest = usize::MAX;

    for direction in HorizontalDir::ALL {
        if direction == from {
            continue;
        }
        let test_pos = offset_pos(pos, direction.offset());
        let Some(test_state) = block_at(region, test_pos) else {
            continue;
        };
        let test_fluid = dynamic_water_state_at(region, test_pos);
        if !can_maybe_pass_through(
            region,
            pos,
            state,
            test_pos,
            test_state,
            test_fluid,
            Some(direction),
        ) {
            continue;
        }
        if is_hole(region, test_pos, test_state) {
            return pass;
        }
        if pass < WATER_SLOPE_FIND_DISTANCE {
            let distance =
                get_slope_distance(region, test_pos, pass + 1, direction.opposite(), test_state);
            lowest = lowest.min(distance);
        }
    }

    lowest
}

fn is_hole(region: &Region, pos: [i32; 3], state: &Block) -> bool {
    let below_pos = offset_pos(pos, [0, -1, 0]);
    let Some(below_state) = block_at(region, below_pos) else {
        return false;
    };
    is_water_hole(region, pos, state, below_pos, below_state)
}

fn is_water_hole(
    region: &Region,
    top_pos: [i32; 3],
    top_state: &Block,
    bottom_pos: [i32; 3],
    bottom_state: &Block,
) -> bool {
    if !can_pass_through_wall_vertical(
        region,
        top_pos,
        top_state,
        bottom_pos,
        bottom_state,
        FaceDirection::Down,
    ) {
        return false;
    }
    dynamic_water_state_at(region, bottom_pos).is_some() || can_hold_specific_fluid(bottom_state)
}

fn can_maybe_pass_through(
    region: &Region,
    source_pos: [i32; 3],
    source_state: &Block,
    target_pos: [i32; 3],
    target_state: &Block,
    target_fluid: Option<DynamicWaterState>,
    direction: Option<HorizontalDir>,
) -> bool {
    !target_fluid.map(|fluid| fluid.is_source()).unwrap_or(false)
        && can_hold_any_fluid(target_state)
        && direction
            .map(|direction| {
                can_pass_through_wall(
                    region,
                    source_pos,
                    source_state,
                    target_pos,
                    target_state,
                    direction,
                )
            })
            .unwrap_or_else(|| {
                can_pass_through_wall_vertical(
                    region,
                    source_pos,
                    source_state,
                    target_pos,
                    target_state,
                    FaceDirection::Down,
                )
            })
}

fn fluid_blocks_motion(block: &Block) -> bool {
    if block.is_air() || dynamic_water_state_at_block(block).is_some() {
        return false;
    }
    if block.namespace != "minecraft" {
        return true;
    }
    if block.id.ends_with("fence_gate") {
        return true;
    }
    blocks_motion(block)
}

fn dynamic_water_state_at_block(block: &Block) -> Option<DynamicWaterState> {
    water_at_single_block(block).map(DynamicWaterState::from_cell)
}

fn water_at_single_block(block: &Block) -> Option<WaterCell> {
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

fn is_waterloggable_block(block: &Block) -> bool {
    block.namespace == "minecraft"
        && (block.id.ends_with("slab")
            || block.id.ends_with("stairs")
            || block.id.ends_with("trapdoor")
            || block.id.ends_with("fence")
            || block.id.ends_with("pane")
            || block.id.ends_with("bars")
            || block.id.ends_with("wall"))
        && !block.id.ends_with("fence_gate")
        && !block.id.ends_with("door")
}

fn slab_is_double(block: &Block) -> bool {
    block.id.ends_with("slab") && block.attributes.get("type").map(String::as_str) == Some("double")
}

fn is_waterlogged(block: &Block) -> bool {
    block.attributes.get("waterlogged").map(String::as_str) == Some("true")
}

fn with_waterlogged(block: &Block, waterlogged: bool) -> Block {
    let mut updated = block.clone();
    updated.attributes.insert(
        "waterlogged".to_string(),
        if waterlogged { "true" } else { "false" }.to_string(),
    );
    updated
}

fn try_place_waterlogged_block(region: &mut Region, pos: [i32; 3], block: &Block) -> bool {
    if !can_place_water_in_block(block) {
        return false;
    }
    let updated = with_waterlogged(block, true);
    let _ = region.set_block(pos, &updated);
    true
}

fn can_place_water_in_block(block: &Block) -> bool {
    is_waterloggable_block(block) && !is_waterlogged(block) && !slab_is_double(block)
}
fn can_hold_any_fluid(block: &Block) -> bool {
    if is_waterloggable_block(block) {
        return true;
    }
    if fluid_blocks_motion(block) {
        return false;
    }
    !matches!(
        block.id.as_str(),
        "ladder"
            | "sugar_cane"
            | "bubble_column"
            | "nether_portal"
            | "end_portal"
            | "end_gateway"
            | "structure_void"
    ) && !block.id.ends_with("door")
        && !block.id.ends_with("sign")
}

fn can_hold_specific_fluid(block: &Block) -> bool {
    !is_waterloggable_block(block) || can_place_water_in_block(block)
}

fn can_pass_through_wall(
    _region: &Region,
    _source_pos: [i32; 3],
    source_state: &Block,
    _target_pos: [i32; 3],
    target_state: &Block,
    direction: HorizontalDir,
) -> bool {
    if is_collision_shape_full_block(source_state) || is_collision_shape_full_block(target_state) {
        return false;
    }
    !merged_face_occludes(source_state, target_state, direction.face())
}

fn can_pass_through_wall_vertical(
    _region: &Region,
    _source_pos: [i32; 3],
    source_state: &Block,
    _target_pos: [i32; 3],
    target_state: &Block,
    direction: FaceDirection,
) -> bool {
    if is_collision_shape_full_block(source_state) || is_collision_shape_full_block(target_state) {
        return false;
    }
    !merged_face_occludes(source_state, target_state, direction)
}

#[derive(Clone, Copy, Debug)]
struct Aabb {
    min_x: f64,
    min_y: f64,
    min_z: f64,
    max_x: f64,
    max_y: f64,
    max_z: f64,
}

#[derive(Clone, Copy, Debug, Default)]
struct MoveResult {
    delta: Vec3d,
    collided_x: bool,
    collided_y: bool,
    collided_z: bool,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct VerifyTickRow {
    tick: usize,
    x: f64,
    y: f64,
    z: f64,
    vx: f64,
    vy: f64,
    vz: f64,
    speed: f64,
    horizontal_speed: f64,
    forward_position: f64,
    forward_velocity: f64,
    lateral_offset: f64,
    on_ground: bool,
    moved: bool,
    collided_x: bool,
    collided_y: bool,
    collided_z: bool,
    no_physics: bool,
    pushout_applied: bool,
    in_water: bool,
    underwater_movement: bool,
    fluid_height: f64,
    current_samples: usize,
    applied_current_x: f64,
    applied_current_y: f64,
    applied_current_z: f64,
    block_x: i32,
    block_y: i32,
    block_z: i32,
    support_block: String,
    center_block: String,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct VerifyMetrics {
    total_ticks: usize,
    target_speed: f64,
    net_displacement_x: f64,
    net_displacement_y: f64,
    net_displacement_z: f64,
    net_horizontal_distance: f64,
    net_forward_distance: f64,
    net_lateral_offset: f64,
    average_horizontal_speed: f64,
    average_forward_speed: f64,
    peak_horizontal_speed: f64,
    mean_horizontal_speed: f64,
    moved_ratio: f64,
    on_ground_ratio: f64,
    in_water_ratio: f64,
    underwater_ratio: f64,
    first_in_water_tick: Option<usize>,
    first_underwater_tick: Option<usize>,
    unsupported_collision_block_count: usize,
    unsupported_collision_blocks: Vec<String>,
    no_physics_ratio: f64,
    pushout_tick_count: usize,
    collision_model: &'static str,
    fluid_model: &'static str,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct VerifyConstantsOutput {
    default_width: f64,
    default_height: f64,
    width: f64,
    height: f64,
    fluid_movement_threshold: f64,
    water_push: f64,
    horizontal_water_damping: f64,
    horizontal_movement_damping: f64,
    vertical_movement_damping: f64,
    gravity: f64,
    buoyancy: f64,
    buoyancy_cap: f64,
    slime_step_on_vy_threshold: f64,
    slime_step_on_base: f64,
    slime_step_on_vy_scale: f64,
    horizontal_rest_threshold2: f64,
    aabb_deflate: f64,
    movement_sample_modulo: usize,
    fluid_current_min_old_movement: f64,
    fluid_current_min_impulse: f64,
    fluid_current_epsilon2: f64,
    no_physics_deflate: f64,
    no_physics_pushout_speed: f64,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct VerifyJsonOutput {
    generated_at: String,
    command: VerifyCommand,
    world_name: String,
    shape: [i32; 3],
    constants: VerifyConstantsOutput,
    metrics: VerifyMetrics,
    inspect_tick: Option<VerifyTickRow>,
}

pub(crate) fn run_verify_command(command: &VerifyCommand) -> Result<(), String> {
    if let Some(tick) = command.inspect_tick {
        if tick > command.ticks {
            return Err("--tick cannot be greater than --ticks.".to_string());
        }
    }

    fs::create_dir_all(&command.out)
        .map_err(|error| format!("Failed to create output dir: {error}"))?;

    let world = load_litematic(&command.input)?;
    let rows = simulate(&world, command);
    let metrics = compute_metrics(&rows, command, &world);
    let inspect_tick = command.inspect_tick.map(|tick| rows[tick].clone());

    let tick_csv_path = command.out.join("verify-ticks.csv");
    let summary_csv_path = command.out.join("verify-summary.csv");
    let summary_json_path = command.out.join("verify-summary.json");

    write_tick_csv(&tick_csv_path, &rows)?;
    write_summary_csv(&summary_csv_path, &metrics, &world)?;
    write_summary_json(
        &summary_json_path,
        command,
        &world,
        &metrics,
        inspect_tick.as_ref(),
    )?;

    println!("World: {}", world.name);
    let shape = world.region.shape();
    println!("Shape: {} x {} x {}", shape[0], shape[1], shape[2]);
    println!("Tick CSV: {}", tick_csv_path.display());
    println!("Summary CSV: {}", summary_csv_path.display());
    println!("Summary JSON: {}", summary_json_path.display());
    if let Some(row) = inspect_tick {
        println!();
        println!(
            "Tick {} => pos=({:.6}, {:.6}, {:.6}) vel=({:.6}, {:.6}, {:.6}) water={} fluidHeight={:.6}",
            row.tick, row.x, row.y, row.z, row.vx, row.vy, row.vz, row.in_water, row.fluid_height
        );
    }
    if !world.approximate_collision_blocks.is_empty() {
        println!();
        println!(
            "Warning: {} block types use approximate non-solid collision handling.",
            world.approximate_collision_blocks.len()
        );
    }
    Ok(())
}

fn simulate(world: &LoadedSchematic, command: &VerifyCommand) -> Vec<VerifyTickRow> {
    let mut region = world.region.clone();
    let mut fluid_ticks = DynamicFluidTicks::bootstrap(&region);
    let mut rows = Vec::with_capacity(command.ticks + 1);
    let start_pos = Vec3d::new(command.start_x, command.start_y, command.start_z);
    let mut pos = start_pos;
    let mut vel = Vec3d::new(command.start_vx, command.start_vy, command.start_vz);
    let mut on_ground = command.start_on_ground;
    let forward_axis = forward_basis(vel);
    let lateral_axis = Vec3d::new(-forward_axis.z, 0.0, forward_axis.x);
    let initial_water = track_water(&region, pos, command.width, command.height, false);
    let initial_no_physics = is_no_physics(&region, pos, command.width, command.height);

    rows.push(build_tick_row(
        0,
        pos,
        vel,
        on_ground,
        false,
        false,
        false,
        false,
        initial_no_physics,
        false,
        initial_water,
        Vec3d::ZERO,
        forward_axis,
        lateral_axis,
        start_pos,
        command.height,
        &region,
    ));

    for tick in 1..=command.ticks {
        fluid_ticks.run_due(&mut region, tick);

        let pre_move_water = track_water(&region, pos, command.width, command.height, false);
        let mut applied_current = if pre_move_water.is_in_water() {
            apply_water_current(&mut vel, pre_move_water)
        } else {
            Vec3d::ZERO
        };

        if pre_move_water.is_in_water() && pre_move_water.applies_underwater_movement() {
            vel.x *= HORIZONTAL_WATER_DAMPING;
            vel.z *= HORIZONTAL_WATER_DAMPING;
            if vel.y < BUOYANCY_CAP {
                vel.y += BUOYANCY;
            }
        } else {
            vel.y -= GRAVITY;
        }

        let no_physics = is_no_physics(&region, pos, command.width, command.height);
        let pushout_applied = if no_physics {
            vel = move_towards_closest_space(&region, pos, vel, command.height);
            true
        } else {
            false
        };

        let phase_mod4 =
            (command.initial_tick_count + tick + command.entity_id_mod4) % MOVEMENT_SAMPLE_MODULO;
        let should_move = !on_ground
            || vel.horizontal_length_sqr() > HORIZONTAL_REST_THRESHOLD2
            || phase_mod4 == 0;
        let mut moved = false;
        let mut collided_x = false;
        let mut collided_y = false;
        let mut collided_z = false;
        let mut actual_on_ground = on_ground;

        if should_move {
            let landed;
            if no_physics {
                pos = pos.add(vel);
                moved = vel.length_sqr() > 1.0e-18;
                actual_on_ground = on_ground;
                landed = false;
            } else {
                let move_result = move_entity(&region, pos, vel, command.width, command.height);
                pos = pos.add(move_result.delta);
                moved = move_result.delta.length_sqr() > 1.0e-18;
                collided_x = move_result.collided_x;
                collided_y = move_result.collided_y;
                collided_z = move_result.collided_z;
                landed = move_result.collided_y && vel.y < 0.0;
                actual_on_ground = landed;
            }
            if landed {
                vel.y =
                    vertical_velocity_after_landing(ground_profile_at(&region, pos).slime, vel.y);
            }
            let drag = horizontal_drag(ground_profile_at(&region, pos), actual_on_ground, vel.y);
            vel.x *= drag;
            vel.z *= drag;
            vel.y *= VERTICAL_MOVEMENT_DAMPING;
            if actual_on_ground && vel.y < 0.0 {
                vel.y *= -0.5;
            }
        }

        let post_move_water = track_water(&region, pos, command.width, command.height, false);
        if post_move_water.is_in_water() {
            applied_current = applied_current.add(apply_water_current(&mut vel, post_move_water));
        }

        rows.push(build_tick_row(
            tick,
            pos,
            vel,
            actual_on_ground,
            moved,
            collided_x,
            collided_y,
            collided_z,
            no_physics,
            pushout_applied,
            post_move_water,
            applied_current,
            forward_axis,
            lateral_axis,
            start_pos,
            command.height,
            &region,
        ));
        on_ground = actual_on_ground;
    }

    rows
}

#[allow(clippy::too_many_arguments)]
fn build_tick_row(
    tick: usize,
    pos: Vec3d,
    vel: Vec3d,
    on_ground: bool,
    moved: bool,
    collided_x: bool,
    collided_y: bool,
    collided_z: bool,
    no_physics: bool,
    pushout_applied: bool,
    water: WaterTracker,
    applied_current: Vec3d,
    forward_axis: Vec3d,
    lateral_axis: Vec3d,
    start_pos: Vec3d,
    entity_height: f64,
    region: &Region,
) -> VerifyTickRow {
    let center_block = centered_block(region, pos, entity_height);
    let support = support_block(region, pos);
    let displacement = Vec3d::new(
        pos.x - start_pos.x,
        pos.y - start_pos.y,
        pos.z - start_pos.z,
    );
    VerifyTickRow {
        tick,
        x: pos.x,
        y: pos.y,
        z: pos.z,
        vx: vel.x,
        vy: vel.y,
        vz: vel.z,
        speed: vel.length(),
        horizontal_speed: vel.horizontal_length(),
        forward_position: displacement.dot_horizontal(forward_axis),
        forward_velocity: vel.dot_horizontal(forward_axis),
        lateral_offset: displacement.dot_horizontal(lateral_axis),
        on_ground,
        moved,
        collided_x,
        collided_y,
        collided_z,
        no_physics,
        pushout_applied,
        in_water: water.is_in_water(),
        underwater_movement: water.applies_underwater_movement(),
        fluid_height: water.height,
        current_samples: water.current_count,
        applied_current_x: applied_current.x,
        applied_current_y: applied_current.y,
        applied_current_z: applied_current.z,
        block_x: pos.x.floor() as i32,
        block_y: pos.y.floor() as i32,
        block_z: pos.z.floor() as i32,
        support_block: support
            .map(block_full_id)
            .unwrap_or_else(|| "minecraft:air".to_string()),
        center_block: center_block
            .map(block_full_id)
            .unwrap_or_else(|| "minecraft:air".to_string()),
    }
}

fn compute_metrics(
    rows: &[VerifyTickRow],
    command: &VerifyCommand,
    world: &LoadedSchematic,
) -> VerifyMetrics {
    let start = &rows[0];
    let end = rows.last().expect("rows must not be empty");
    let tick_count = command.ticks;
    let tick_denominator = tick_count.max(1) as f64;
    let mut peak_horizontal_speed: f64 = 0.0;
    let mut horizontal_speed_sum = 0.0;
    let mut moved_count = 0_usize;
    let mut on_ground_count = 0_usize;
    let mut in_water_count = 0_usize;
    let mut underwater_count = 0_usize;
    let mut no_physics_count = 0_usize;
    let mut pushout_tick_count = 0_usize;
    let mut first_in_water_tick = None;
    let mut first_underwater_tick = None;

    for row in rows.iter().skip(1) {
        peak_horizontal_speed = peak_horizontal_speed.max(row.horizontal_speed);
        horizontal_speed_sum += row.horizontal_speed;
        moved_count += usize::from(row.moved);
        on_ground_count += usize::from(row.on_ground);
        in_water_count += usize::from(row.in_water);
        underwater_count += usize::from(row.underwater_movement);
        no_physics_count += usize::from(row.no_physics);
        pushout_tick_count += usize::from(row.pushout_applied);
        if first_in_water_tick.is_none() && row.in_water {
            first_in_water_tick = Some(row.tick);
        }
        if first_underwater_tick.is_none() && row.underwater_movement {
            first_underwater_tick = Some(row.tick);
        }
    }

    let net_displacement_x = end.x - start.x;
    let net_displacement_y = end.y - start.y;
    let net_displacement_z = end.z - start.z;
    let net_horizontal_distance =
        (net_displacement_x * net_displacement_x + net_displacement_z * net_displacement_z).sqrt();

    VerifyMetrics {
        total_ticks: tick_count,
        target_speed: command.target_speed,
        net_displacement_x,
        net_displacement_y,
        net_displacement_z,
        net_horizontal_distance,
        net_forward_distance: end.forward_position,
        net_lateral_offset: end.lateral_offset,
        average_horizontal_speed: net_horizontal_distance / tick_denominator,
        average_forward_speed: end.forward_position / tick_denominator,
        peak_horizontal_speed,
        mean_horizontal_speed: horizontal_speed_sum / tick_denominator,
        moved_ratio: moved_count as f64 / tick_denominator,
        on_ground_ratio: on_ground_count as f64 / tick_denominator,
        in_water_ratio: in_water_count as f64 / tick_denominator,
        underwater_ratio: underwater_count as f64 / tick_denominator,
        first_in_water_tick,
        first_underwater_tick,
        unsupported_collision_block_count: world.approximate_collision_blocks.len(),
        unsupported_collision_blocks: world.approximate_collision_blocks.clone(),
        no_physics_ratio: no_physics_count as f64 / tick_denominator,
        pushout_tick_count,
        collision_model: VERIFY_COLLISION_MODEL,
        fluid_model: VERIFY_FLUID_MODEL,
    }
}

fn write_tick_csv(path: &Path, rows: &[VerifyTickRow]) -> Result<(), String> {
    let columns = [
        "tick",
        "x",
        "y",
        "z",
        "vx",
        "vy",
        "vz",
        "speed",
        "horizontalSpeed",
        "forwardPosition",
        "forwardVelocity",
        "lateralOffset",
        "onGround",
        "moved",
        "collidedX",
        "collidedY",
        "collidedZ",
        "noPhysics",
        "pushoutApplied",
        "inWater",
        "underwaterMovement",
        "fluidHeight",
        "currentSamples",
        "appliedCurrentX",
        "appliedCurrentY",
        "appliedCurrentZ",
        "blockX",
        "blockY",
        "blockZ",
        "supportBlock",
        "centerBlock",
    ];
    let mut lines = Vec::with_capacity(rows.len() + 1);
    lines.push(columns.join(","));
    for row in rows {
        let values = columns
            .iter()
            .map(|column| csv_escape(&tick_csv_value(row, column)))
            .collect::<Vec<_>>();
        lines.push(values.join(","));
    }
    fs::write(path, format!("{}\n", lines.join("\n")))
        .map_err(|error| format!("Failed to write verify tick CSV: {error}"))
}

fn write_summary_csv(
    path: &Path,
    metrics: &VerifyMetrics,
    world: &LoadedSchematic,
) -> Result<(), String> {
    let shape = world.region.shape();
    let columns = [
        "worldName",
        "sizeX",
        "sizeY",
        "sizeZ",
        "totalTicks",
        "targetSpeed",
        "netDisplacementX",
        "netDisplacementY",
        "netDisplacementZ",
        "netHorizontalDistance",
        "netForwardDistance",
        "netLateralOffset",
        "averageHorizontalSpeed",
        "averageForwardSpeed",
        "peakHorizontalSpeed",
        "meanHorizontalSpeed",
        "movedRatio",
        "onGroundRatio",
        "inWaterRatio",
        "underwaterRatio",
        "firstInWaterTick",
        "firstUnderwaterTick",
        "unsupportedCollisionBlockCount",
        "unsupportedCollisionBlocks",
        "noPhysicsRatio",
        "pushoutTickCount",
        "collisionModel",
        "fluidModel",
    ];
    let values = [
        world.name.clone(),
        shape[0].to_string(),
        shape[1].to_string(),
        shape[2].to_string(),
        metrics.total_ticks.to_string(),
        metrics.target_speed.to_string(),
        metrics.net_displacement_x.to_string(),
        metrics.net_displacement_y.to_string(),
        metrics.net_displacement_z.to_string(),
        metrics.net_horizontal_distance.to_string(),
        metrics.net_forward_distance.to_string(),
        metrics.net_lateral_offset.to_string(),
        metrics.average_horizontal_speed.to_string(),
        metrics.average_forward_speed.to_string(),
        metrics.peak_horizontal_speed.to_string(),
        metrics.mean_horizontal_speed.to_string(),
        metrics.moved_ratio.to_string(),
        metrics.on_ground_ratio.to_string(),
        metrics.in_water_ratio.to_string(),
        metrics.underwater_ratio.to_string(),
        option_usize(metrics.first_in_water_tick),
        option_usize(metrics.first_underwater_tick),
        metrics.unsupported_collision_block_count.to_string(),
        metrics.unsupported_collision_blocks.join(";"),
        metrics.no_physics_ratio.to_string(),
        metrics.pushout_tick_count.to_string(),
        metrics.collision_model.to_string(),
        metrics.fluid_model.to_string(),
    ];
    let header = columns.join(",");
    let row = values
        .iter()
        .map(|value| csv_escape(value))
        .collect::<Vec<_>>()
        .join(",");
    fs::write(path, format!("{header}\n{row}\n"))
        .map_err(|error| format!("Failed to write verify summary CSV: {error}"))
}

fn write_summary_json(
    path: &Path,
    command: &VerifyCommand,
    world: &LoadedSchematic,
    metrics: &VerifyMetrics,
    inspect_tick: Option<&VerifyTickRow>,
) -> Result<(), String> {
    let generated_at = OffsetDateTime::now_utc()
        .format(&Rfc3339)
        .map_err(|error| format!("Failed to format timestamp: {error}"))?;
    let output = VerifyJsonOutput {
        generated_at,
        command: command.clone(),
        world_name: world.name.clone(),
        shape: world.region.shape(),
        constants: VerifyConstantsOutput {
            default_width: VERIFY_DEFAULT_WIDTH,
            default_height: VERIFY_DEFAULT_HEIGHT,
            width: command.width,
            height: command.height,
            fluid_movement_threshold: FLUID_MOVEMENT_THRESHOLD,
            water_push: WATER_PUSH,
            horizontal_water_damping: HORIZONTAL_WATER_DAMPING,
            horizontal_movement_damping: HORIZONTAL_MOVEMENT_DAMPING,
            vertical_movement_damping: VERTICAL_MOVEMENT_DAMPING,
            gravity: GRAVITY,
            buoyancy: BUOYANCY,
            buoyancy_cap: BUOYANCY_CAP,
            slime_step_on_vy_threshold: SLIME_STEP_ON_VY_THRESHOLD,
            slime_step_on_base: SLIME_STEP_ON_BASE,
            slime_step_on_vy_scale: SLIME_STEP_ON_VY_SCALE,
            horizontal_rest_threshold2: HORIZONTAL_REST_THRESHOLD2,
            aabb_deflate: AABB_DEFLATE,
            movement_sample_modulo: MOVEMENT_SAMPLE_MODULO,
            fluid_current_min_old_movement: FLUID_CURRENT_MIN_OLD_MOVEMENT,
            fluid_current_min_impulse: FLUID_CURRENT_MIN_IMPULSE,
            fluid_current_epsilon2: FLUID_CURRENT_EPSILON2,
            no_physics_deflate: NO_PHYSICS_DEFLATE,
            no_physics_pushout_speed: NO_PHYSICS_PUSHOUT_SPEED,
        },
        metrics: metrics.clone(),
        inspect_tick: inspect_tick.cloned(),
    };
    let json = serde_json::to_string_pretty(&output)
        .map_err(|error| format!("Failed to serialize verify JSON: {error}"))?;
    fs::write(path, format!("{json}\n"))
        .map_err(|error| format!("Failed to write verify summary JSON: {error}"))
}

fn tick_csv_value(row: &VerifyTickRow, column: &str) -> String {
    match column {
        "tick" => row.tick.to_string(),
        "x" => row.x.to_string(),
        "y" => row.y.to_string(),
        "z" => row.z.to_string(),
        "vx" => row.vx.to_string(),
        "vy" => row.vy.to_string(),
        "vz" => row.vz.to_string(),
        "speed" => row.speed.to_string(),
        "horizontalSpeed" => row.horizontal_speed.to_string(),
        "forwardPosition" => row.forward_position.to_string(),
        "forwardVelocity" => row.forward_velocity.to_string(),
        "lateralOffset" => row.lateral_offset.to_string(),
        "onGround" => row.on_ground.to_string(),
        "moved" => row.moved.to_string(),
        "collidedX" => row.collided_x.to_string(),
        "collidedY" => row.collided_y.to_string(),
        "collidedZ" => row.collided_z.to_string(),
        "noPhysics" => row.no_physics.to_string(),
        "pushoutApplied" => row.pushout_applied.to_string(),
        "inWater" => row.in_water.to_string(),
        "underwaterMovement" => row.underwater_movement.to_string(),
        "fluidHeight" => row.fluid_height.to_string(),
        "currentSamples" => row.current_samples.to_string(),
        "appliedCurrentX" => row.applied_current_x.to_string(),
        "appliedCurrentY" => row.applied_current_y.to_string(),
        "appliedCurrentZ" => row.applied_current_z.to_string(),
        "blockX" => row.block_x.to_string(),
        "blockY" => row.block_y.to_string(),
        "blockZ" => row.block_z.to_string(),
        "supportBlock" => row.support_block.clone(),
        "centerBlock" => row.center_block.clone(),
        _ => String::new(),
    }
}

fn csv_escape(value: &str) -> String {
    if value
        .chars()
        .any(|ch| matches!(ch, '"' | ',' | '\r' | '\n'))
    {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

fn option_usize(value: Option<usize>) -> String {
    value.map(|value| value.to_string()).unwrap_or_default()
}

fn forward_basis(velocity: Vec3d) -> Vec3d {
    let horizontal = Vec3d::new(velocity.x, 0.0, velocity.z);
    if horizontal.horizontal_length_sqr() <= 1.0e-12 {
        Vec3d::new(1.0, 0.0, 0.0)
    } else {
        horizontal.normalized()
    }
}

fn centered_block(region: &Region, pos: Vec3d, height: f64) -> Option<&Block> {
    block_at(
        region,
        [
            pos.x.floor() as i32,
            (pos.y + height * 0.5).floor() as i32,
            pos.z.floor() as i32,
        ],
    )
}

fn support_block(region: &Region, pos: Vec3d) -> Option<&Block> {
    block_at(
        region,
        [
            pos.x.floor() as i32,
            (pos.y - 1.0e-6).floor() as i32,
            pos.z.floor() as i32,
        ],
    )
}

fn ground_profile_at(region: &Region, pos: Vec3d) -> GroundProfile {
    let Some(block) = support_block(region, pos) else {
        return GroundProfile {
            friction: 0.6_f32 as f64,
            slime: false,
        };
    };
    let friction = if block.namespace == "minecraft" {
        match block.id.as_str() {
            "packed_ice" | "ice" | "frosted_ice" => 0.98_f32 as f64,
            "blue_ice" => 0.989_f32 as f64,
            "slime_block" => 0.8_f32 as f64,
            _ => 0.6_f32 as f64,
        }
    } else {
        0.6_f32 as f64
    };
    GroundProfile {
        friction,
        slime: block.namespace == "minecraft" && block.id == "slime_block",
    }
}

fn horizontal_drag(profile: GroundProfile, on_ground: bool, vy: f64) -> f64 {
    let mut drag = HORIZONTAL_MOVEMENT_DAMPING;
    if on_ground {
        drag = profile.friction * HORIZONTAL_MOVEMENT_DAMPING;
        if profile.slime && vy.abs() < SLIME_STEP_ON_VY_THRESHOLD {
            drag *= SLIME_STEP_ON_BASE + SLIME_STEP_ON_VY_SCALE * vy.abs();
        }
    }
    drag
}

fn vertical_velocity_after_landing(on_slime: bool, vy: f64) -> f64 {
    if on_slime && vy < 0.0 {
        -vy * 0.8
    } else if on_slime {
        vy
    } else {
        0.0
    }
}

fn track_water(
    region: &Region,
    pos: Vec3d,
    width: f64,
    height: f64,
    ignore_current: bool,
) -> WaterTracker {
    let half_width = width * 0.5;
    let box_min_x = pos.x - half_width + AABB_DEFLATE;
    let box_max_x = pos.x + half_width - AABB_DEFLATE;
    let box_min_y = pos.y + AABB_DEFLATE;
    let box_max_y = pos.y + height - AABB_DEFLATE;
    let box_min_z = pos.z - half_width + AABB_DEFLATE;
    let box_max_z = pos.z + half_width - AABB_DEFLATE;
    let x0 = box_min_x.floor() as i32;
    let x1 = box_max_x.ceil() as i32 - 1;
    let y0 = box_min_y.floor() as i32;
    let y1 = box_max_y.ceil() as i32 - 1;
    let z0 = box_min_z.floor() as i32;
    let z1 = box_max_z.ceil() as i32 - 1;
    let mut tracker = WaterTracker::default();

    for x in x0..=x1 {
        for y in y0..=y1 {
            for z in z0..=z1 {
                let cell_pos = [x, y, z];
                let Some(water) = water_at(region, cell_pos) else {
                    continue;
                };
                let fluid_top = y as f64 + water.height;
                if fluid_top < box_min_y {
                    continue;
                }
                tracker.height = tracker.height.max(fluid_top - pos.y);
                if !ignore_current {
                    let mut flow = fluid_flow(region, cell_pos, water);
                    if tracker.height < 0.4 {
                        flow = flow.scale(tracker.height);
                    }
                    tracker.accumulated_current = tracker.accumulated_current.add(flow);
                    tracker.current_count += 1;
                }
            }
        }
    }

    tracker
}

fn fluid_flow(region: &Region, pos: [i32; 3], water: WaterCell) -> Vec3d {
    let mut flow_x = 0.0;
    let mut flow_z = 0.0;
    for (dx, dz) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
        let neighbor_pos = [pos[0] + dx, pos[1], pos[2] + dz];
        let neighbor_water = water_at(region, neighbor_pos);
        let mut neighbor_height = neighbor_water.map(|cell| cell.own_height).unwrap_or(0.0);
        let mut distance = 0.0;
        if neighbor_height == 0.0 {
            let neighbor_block = block_at(region, neighbor_pos);
            if neighbor_block
                .map(|block| !fluid_blocks_motion(block))
                .unwrap_or(true)
            {
                let below_neighbor = water_at(
                    region,
                    [neighbor_pos[0], neighbor_pos[1] - 1, neighbor_pos[2]],
                );
                if let Some(below_neighbor) = below_neighbor {
                    neighbor_height = below_neighbor.own_height;
                    if neighbor_height > 0.0 {
                        distance = water.own_height - (neighbor_height - 0.888_888_9_f32 as f64);
                    }
                }
            }
        } else {
            distance = water.own_height - neighbor_height;
        }
        if distance != 0.0 {
            flow_x += dx as f64 * distance;
            flow_z += dz as f64 * distance;
        }
    }

    let mut flow = Vec3d::new(flow_x, 0.0, flow_z).normalized();
    if water.falling {
        for (dx, dz, direction) in [
            (-1, 0, FaceDirection::West),
            (1, 0, FaceDirection::East),
            (0, -1, FaceDirection::North),
            (0, 1, FaceDirection::South),
        ] {
            let neighbor = [pos[0] + dx, pos[1], pos[2] + dz];
            let above_neighbor = [neighbor[0], neighbor[1] + 1, neighbor[2]];
            if is_solid_face(region, neighbor, direction)
                || is_solid_face(region, above_neighbor, direction)
            {
                flow = flow
                    .normalized()
                    .add(Vec3d::new(0.0, -6.0, 0.0))
                    .normalized();
                break;
            }
        }
    }
    flow
}

fn is_solid_face(region: &Region, pos: [i32; 3], direction: FaceDirection) -> bool {
    let Some(block) = block_at(region, pos) else {
        return false;
    };
    is_face_sturdy(block, direction)
}

fn apply_water_current(velocity: &mut Vec3d, tracker: WaterTracker) -> Vec3d {
    if tracker.current_count == 0
        || tracker.accumulated_current.length_sqr() < FLUID_CURRENT_EPSILON2
    {
        return Vec3d::ZERO;
    }

    let mut impulse = tracker.accumulated_current.normalized().scale(WATER_PUSH);
    if velocity.x.abs() < FLUID_CURRENT_MIN_OLD_MOVEMENT
        && velocity.z.abs() < FLUID_CURRENT_MIN_OLD_MOVEMENT
        && impulse.length() < FLUID_CURRENT_MIN_IMPULSE
    {
        impulse = impulse.normalized().scale(FLUID_CURRENT_MIN_IMPULSE);
    }
    *velocity = velocity.add(impulse);
    impulse
}

fn move_entity(
    region: &Region,
    pos: Vec3d,
    velocity: Vec3d,
    width: f64,
    height: f64,
) -> MoveResult {
    let mut aabb = entity_aabb(pos, width, height);
    let (dy, collided_y) = collide_axis(region, aabb, velocity.y, Axis::Y);
    aabb = shift_aabb(aabb, Axis::Y, dy);

    let mut dx = 0.0;
    let mut dz = 0.0;
    let mut collided_x = false;
    let mut collided_z = false;
    for axis in horizontal_axes_in_order(velocity) {
        let (resolved, collided) = collide_axis(region, aabb, axis_component(velocity, axis), axis);
        aabb = shift_aabb(aabb, axis, resolved);
        match axis {
            Axis::X => {
                dx = resolved;
                collided_x = collided;
            }
            Axis::Z => {
                dz = resolved;
                collided_z = collided;
            }
            Axis::Y => unreachable!("vertical axis is resolved first"),
        }
    }

    MoveResult {
        delta: Vec3d::new(dx, dy, dz),
        collided_x,
        collided_y,
        collided_z,
    }
}

fn horizontal_axes_in_order(velocity: Vec3d) -> [Axis; 2] {
    if velocity.x.abs() < velocity.z.abs() {
        [Axis::Z, Axis::X]
    } else {
        [Axis::X, Axis::Z]
    }
}

fn axis_component(velocity: Vec3d, axis: Axis) -> f64 {
    match axis {
        Axis::X => velocity.x,
        Axis::Y => velocity.y,
        Axis::Z => velocity.z,
    }
}

fn is_no_physics(region: &Region, pos: Vec3d, width: f64, height: f64) -> bool {
    aabb_intersects_world(
        region,
        deflate_aabb(entity_aabb(pos, width, height), NO_PHYSICS_DEFLATE),
    )
}

fn move_towards_closest_space(region: &Region, pos: Vec3d, velocity: Vec3d, height: f64) -> Vec3d {
    let center_y = pos.y + height * 0.5;
    let block_x = pos.x.floor() as i32;
    let block_y = center_y.floor() as i32;
    let block_z = pos.z.floor() as i32;
    let delta_x = pos.x - block_x as f64;
    let delta_y = center_y - block_y as f64;
    let delta_z = pos.z - block_z as f64;
    let mut closest_axis = Axis::Y;
    let mut closest_step = 1.0;
    let mut closest = f64::INFINITY;

    for (neighbor_pos, axis, step, oriented_delta) in [
        ([block_x, block_y, block_z - 1], Axis::Z, -1.0, delta_z),
        ([block_x, block_y, block_z + 1], Axis::Z, 1.0, 1.0 - delta_z),
        ([block_x - 1, block_y, block_z], Axis::X, -1.0, delta_x),
        ([block_x + 1, block_y, block_z], Axis::X, 1.0, 1.0 - delta_x),
        ([block_x, block_y + 1, block_z], Axis::Y, 1.0, 1.0 - delta_y),
    ] {
        if !is_collision_shape_full_block_at(region, neighbor_pos) && oriented_delta < closest {
            closest = oriented_delta;
            closest_axis = axis;
            closest_step = step;
        }
    }

    let scaled = velocity.scale(0.75);
    match closest_axis {
        Axis::X => Vec3d::new(closest_step * NO_PHYSICS_PUSHOUT_SPEED, scaled.y, scaled.z),
        Axis::Y => Vec3d::new(scaled.x, closest_step * NO_PHYSICS_PUSHOUT_SPEED, scaled.z),
        Axis::Z => Vec3d::new(scaled.x, scaled.y, closest_step * NO_PHYSICS_PUSHOUT_SPEED),
    }
}

fn is_collision_shape_full_block_at(region: &Region, pos: [i32; 3]) -> bool {
    block_at(region, pos)
        .map(is_collision_shape_full_block)
        .unwrap_or(false)
}

#[derive(Clone, Copy, Debug)]
enum Axis {
    X,
    Y,
    Z,
}

fn deflate_aabb(aabb: Aabb, amount: f64) -> Aabb {
    Aabb {
        min_x: aabb.min_x + amount,
        min_y: aabb.min_y + amount,
        min_z: aabb.min_z + amount,
        max_x: aabb.max_x - amount,
        max_y: aabb.max_y - amount,
        max_z: aabb.max_z - amount,
    }
}

fn aabb_intersects_world(region: &Region, aabb: Aabb) -> bool {
    let x0 = aabb.min_x.floor() as i32;
    let x1 = aabb.max_x.ceil() as i32 - 1;
    let y0 = aabb.min_y.floor() as i32;
    let y1 = aabb.max_y.ceil() as i32 - 1;
    let z0 = aabb.min_z.floor() as i32;
    let z1 = aabb.max_z.ceil() as i32 - 1;

    for x in x0..=x1 {
        for y in y0..=y1 {
            for z in z0..=z1 {
                let Some(block) = block_at(region, [x, y, z]) else {
                    continue;
                };
                for collision_box in collision_boxes(block).iter() {
                    if aabbs_intersect(aabb, world_collision_box([x, y, z], collision_box)) {
                        return true;
                    }
                }
            }
        }
    }
    false
}

fn aabbs_intersect(left: Aabb, right: Aabb) -> bool {
    left.max_x > right.min_x + 1.0e-12
        && left.min_x < right.max_x - 1.0e-12
        && left.max_y > right.min_y + 1.0e-12
        && left.min_y < right.max_y - 1.0e-12
        && left.max_z > right.min_z + 1.0e-12
        && left.min_z < right.max_z - 1.0e-12
}

fn entity_aabb(pos: Vec3d, width: f64, height: f64) -> Aabb {
    let half_width = width * 0.5;
    Aabb {
        min_x: pos.x - half_width,
        min_y: pos.y,
        min_z: pos.z - half_width,
        max_x: pos.x + half_width,
        max_y: pos.y + height,
        max_z: pos.z + half_width,
    }
}

fn shift_aabb(aabb: Aabb, axis: Axis, delta: f64) -> Aabb {
    match axis {
        Axis::X => Aabb {
            min_x: aabb.min_x + delta,
            max_x: aabb.max_x + delta,
            ..aabb
        },
        Axis::Y => Aabb {
            min_y: aabb.min_y + delta,
            max_y: aabb.max_y + delta,
            ..aabb
        },
        Axis::Z => Aabb {
            min_z: aabb.min_z + delta,
            max_z: aabb.max_z + delta,
            ..aabb
        },
    }
}

fn world_collision_box(block_pos: [i32; 3], collision_box: &CollisionBox) -> Aabb {
    Aabb {
        min_x: block_pos[0] as f64 + collision_box.min_x,
        min_y: block_pos[1] as f64 + collision_box.min_y,
        min_z: block_pos[2] as f64 + collision_box.min_z,
        max_x: block_pos[0] as f64 + collision_box.max_x,
        max_y: block_pos[1] as f64 + collision_box.max_y,
        max_z: block_pos[2] as f64 + collision_box.max_z,
    }
}

fn collide_axis(region: &Region, aabb: Aabb, delta: f64, axis: Axis) -> (f64, bool) {
    if delta == 0.0 {
        return (0.0, false);
    }

    let (scan_x0, scan_x1, scan_y0, scan_y1, scan_z0, scan_z1) = match axis {
        Axis::X => (
            (if delta > 0.0 {
                aabb.max_x
            } else {
                aabb.min_x + delta
            })
            .floor() as i32,
            (if delta > 0.0 {
                aabb.max_x + delta
            } else {
                aabb.min_x
            })
            .ceil() as i32
                - 1,
            aabb.min_y.floor() as i32,
            aabb.max_y.ceil() as i32 - 1,
            aabb.min_z.floor() as i32,
            aabb.max_z.ceil() as i32 - 1,
        ),
        Axis::Y => (
            aabb.min_x.floor() as i32,
            aabb.max_x.ceil() as i32 - 1,
            (if delta > 0.0 {
                aabb.max_y
            } else {
                aabb.min_y + delta
            })
            .floor() as i32,
            (if delta > 0.0 {
                aabb.max_y + delta
            } else {
                aabb.min_y
            })
            .ceil() as i32
                - 1,
            aabb.min_z.floor() as i32,
            aabb.max_z.ceil() as i32 - 1,
        ),
        Axis::Z => (
            aabb.min_x.floor() as i32,
            aabb.max_x.ceil() as i32 - 1,
            aabb.min_y.floor() as i32,
            aabb.max_y.ceil() as i32 - 1,
            (if delta > 0.0 {
                aabb.max_z
            } else {
                aabb.min_z + delta
            })
            .floor() as i32,
            (if delta > 0.0 {
                aabb.max_z + delta
            } else {
                aabb.min_z
            })
            .ceil() as i32
                - 1,
        ),
    };

    let mut allowed = delta;
    let mut collided = false;
    for x in scan_x0..=scan_x1 {
        for y in scan_y0..=scan_y1 {
            for z in scan_z0..=scan_z1 {
                let Some(block) = block_at(region, [x, y, z]) else {
                    continue;
                };
                for collision_box in collision_boxes(block).iter() {
                    let world_box = world_collision_box([x, y, z], collision_box);
                    match axis {
                        Axis::X => {
                            if !ranges_overlap(
                                aabb.min_y,
                                aabb.max_y,
                                world_box.min_y,
                                world_box.max_y,
                            ) || !ranges_overlap(
                                aabb.min_z,
                                aabb.max_z,
                                world_box.min_z,
                                world_box.max_z,
                            ) {
                                continue;
                            }
                            if allowed > 0.0 {
                                let gap = world_box.min_x - aabb.max_x;
                                if gap >= 0.0 && gap < allowed {
                                    allowed = gap;
                                    collided = true;
                                }
                            } else {
                                let gap = world_box.max_x - aabb.min_x;
                                if gap <= 0.0 && gap > allowed {
                                    allowed = gap;
                                    collided = true;
                                }
                            }
                        }
                        Axis::Y => {
                            if !ranges_overlap(
                                aabb.min_x,
                                aabb.max_x,
                                world_box.min_x,
                                world_box.max_x,
                            ) || !ranges_overlap(
                                aabb.min_z,
                                aabb.max_z,
                                world_box.min_z,
                                world_box.max_z,
                            ) {
                                continue;
                            }
                            if allowed > 0.0 {
                                let gap = world_box.min_y - aabb.max_y;
                                if gap >= 0.0 && gap < allowed {
                                    allowed = gap;
                                    collided = true;
                                }
                            } else {
                                let gap = world_box.max_y - aabb.min_y;
                                if gap <= 0.0 && gap > allowed {
                                    allowed = gap;
                                    collided = true;
                                }
                            }
                        }
                        Axis::Z => {
                            if !ranges_overlap(
                                aabb.min_x,
                                aabb.max_x,
                                world_box.min_x,
                                world_box.max_x,
                            ) || !ranges_overlap(
                                aabb.min_y,
                                aabb.max_y,
                                world_box.min_y,
                                world_box.max_y,
                            ) {
                                continue;
                            }
                            if allowed > 0.0 {
                                let gap = world_box.min_z - aabb.max_z;
                                if gap >= 0.0 && gap < allowed {
                                    allowed = gap;
                                    collided = true;
                                }
                            } else {
                                let gap = world_box.max_z - aabb.min_z;
                                if gap <= 0.0 && gap > allowed {
                                    allowed = gap;
                                    collided = true;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    (allowed, collided)
}

fn ranges_overlap(a_min: f64, a_max: f64, b_min: f64, b_max: f64) -> bool {
    a_max > b_min + 1.0e-12 && a_min < b_max - 1.0e-12
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_block(id: &str) -> Block {
        Block::from_id(id).expect("valid block")
    }

    fn region_with_shape(shape: [i32; 3]) -> Region {
        Region::with_shape(shape)
    }

    fn simple_channel_world() -> LoadedSchematic {
        let mut region = region_with_shape([14, 5, 3]);
        region.name = "simple_channel".to_string();

        for x in 1..=13 {
            region
                .set_block([x, 0, 1], &parse_block("minecraft:packed_ice"))
                .unwrap();
            for y in 1..=3 {
                region
                    .set_block([x, y, 0], &parse_block("minecraft:glass"))
                    .unwrap();
                region
                    .set_block([x, y, 2], &parse_block("minecraft:glass"))
                    .unwrap();
            }
            for z in 0..=2 {
                region
                    .set_block([x, 4, z], &parse_block("minecraft:glass"))
                    .unwrap();
            }
        }

        for y in 1..=3 {
            for z in 0..=2 {
                region
                    .set_block([0, y, z], &parse_block("minecraft:glass"))
                    .unwrap();
            }
        }

        region
            .set_block([1, 1, 1], &parse_block("minecraft:water"))
            .unwrap();
        for (x, level) in [(2, 1), (3, 2), (4, 3), (5, 4), (6, 5), (7, 6), (8, 7)] {
            region
                .set_block(
                    [x, 1, 1],
                    &parse_block(&format!("minecraft:water[level={level}]")),
                )
                .unwrap();
        }

        LoadedSchematic {
            name: "simple-channel".to_string(),
            region,
            approximate_collision_blocks: Vec::new(),
        }
    }

    fn complex_static_gate_world() -> LoadedSchematic {
        let mut region = region_with_shape([26, 5, 3]);
        region.name = "complex_static_gate".to_string();

        for x in 0..=24 {
            let floor = if x % 4 == 0 {
                "minecraft:packed_ice"
            } else {
                "minecraft:blue_ice"
            };
            region.set_block([x, 0, 1], &parse_block(floor)).unwrap();
            region
                .set_block([x, 0, 0], &parse_block("minecraft:smooth_stone"))
                .unwrap();
            region
                .set_block([x, 0, 2], &parse_block("minecraft:smooth_stone"))
                .unwrap();
            for y in 1..=3 {
                region
                    .set_block([x, y, 0], &parse_block("minecraft:glass"))
                    .unwrap();
                region
                    .set_block([x, y, 2], &parse_block("minecraft:glass"))
                    .unwrap();
            }
            for z in 0..=2 {
                region
                    .set_block([x, 4, z], &parse_block("minecraft:glass"))
                    .unwrap();
            }
        }

        for (x, block) in [
            (0, "minecraft:water"),
            (1, "minecraft:water[level=1]"),
            (2, "minecraft:water[level=2]"),
            (3, "minecraft:water[level=3]"),
            (4, "minecraft:oak_fence_gate[facing=north,open=true]"),
            (5, "minecraft:water[level=5]"),
            (6, "minecraft:water[level=6]"),
            (7, "minecraft:water[level=7]"),
            (10, "minecraft:water"),
            (11, "minecraft:water[level=1]"),
            (12, "minecraft:water[level=2]"),
            (13, "minecraft:water[level=3]"),
            (14, "minecraft:oak_fence_gate[facing=north,open=true]"),
            (15, "minecraft:water[level=5]"),
            (16, "minecraft:water[level=6]"),
            (17, "minecraft:water[level=7]"),
            (20, "minecraft:water"),
            (21, "minecraft:water[level=1]"),
            (22, "minecraft:water[level=2]"),
            (23, "minecraft:water[level=3]"),
            (24, "minecraft:oak_fence_gate[facing=north,open=true]"),
        ] {
            region.set_block([x, 1, 1], &parse_block(block)).unwrap();
        }

        LoadedSchematic {
            name: "complex-static-gate".to_string(),
            region,
            approximate_collision_blocks: Vec::new(),
        }
    }

    fn waterlogged_static_world() -> LoadedSchematic {
        let mut region = region_with_shape([14, 5, 3]);
        region.name = "waterlogged_static".to_string();

        for x in 1..=12 {
            region
                .set_block([x, 0, 0], &parse_block("minecraft:smooth_stone"))
                .unwrap();
            region
                .set_block([x, 0, 1], &parse_block("minecraft:smooth_stone"))
                .unwrap();
            region
                .set_block([x, 0, 2], &parse_block("minecraft:smooth_stone"))
                .unwrap();
            for y in 1..=3 {
                region
                    .set_block([x, y, 0], &parse_block("minecraft:glass"))
                    .unwrap();
                region
                    .set_block([x, y, 2], &parse_block("minecraft:glass"))
                    .unwrap();
            }
            for z in 0..=2 {
                region
                    .set_block([x, 4, z], &parse_block("minecraft:glass"))
                    .unwrap();
            }
        }

        for y in 1..=3 {
            for z in 0..=2 {
                region
                    .set_block([0, y, z], &parse_block("minecraft:glass"))
                    .unwrap();
                region
                    .set_block([13, y, z], &parse_block("minecraft:glass"))
                    .unwrap();
            }
        }

        for (x, block) in [
            (1, "minecraft:water"),
            (2, "minecraft:oak_slab[type=bottom]"),
            (4, "minecraft:cobblestone_wall"),
            (6, "minecraft:iron_bars"),
            (
                8,
                "minecraft:oak_trapdoor[half=bottom,open=true,facing=north]",
            ),
            (
                10,
                "minecraft:oak_stairs[facing=east,half=bottom,shape=straight]",
            ),
            (11, "minecraft:water[level=1]"),
            (12, "minecraft:water"),
        ] {
            region.set_block([x, 1, 1], &parse_block(block)).unwrap();
        }

        LoadedSchematic {
            name: "waterlogged-static".to_string(),
            region,
            approximate_collision_blocks: Vec::new(),
        }
    }

    fn run_fluid_ticks(region: &mut Region, ticks: usize) {
        let mut fluid_ticks = DynamicFluidTicks::bootstrap(region);
        for tick in 1..=ticks {
            fluid_ticks.run_due(region, tick);
        }
    }

    #[test]
    fn water_flow_tracks_gradient_direction() {
        let mut region = region_with_shape([2, 1, 1]);
        region
            .set_block([0, 0, 0], &parse_block("minecraft:water[level=1]"))
            .unwrap();
        region
            .set_block([1, 0, 0], &parse_block("minecraft:water[level=6]"))
            .unwrap();
        let flow = fluid_flow(
            &region,
            [0, 0, 0],
            water_at(&region, [0, 0, 0]).expect("water"),
        );
        assert!(flow.x > 0.0);
        assert!(flow.z.abs() < 1.0e-12);
    }

    #[test]
    fn collisions_stop_at_full_blocks() {
        let mut region = region_with_shape([3, 2, 1]);
        region
            .set_block([1, 0, 0], &parse_block("minecraft:stone"))
            .unwrap();
        region
            .set_block([1, 1, 0], &parse_block("minecraft:stone"))
            .unwrap();
        let move_result = move_entity(
            &region,
            Vec3d::new(0.5, 0.0, 0.5),
            Vec3d::new(1.0, 0.0, 0.0),
            0.25,
            0.25,
        );
        assert!(move_result.collided_x);
        assert!(move_result.delta.x < 1.0);
    }

    #[test]
    fn collisions_stop_on_bottom_slabs() {
        let mut region = region_with_shape([1, 1, 1]);
        region
            .set_block([0, 0, 0], &parse_block("minecraft:oak_slab[type=bottom]"))
            .unwrap();
        let move_result = move_entity(
            &region,
            Vec3d::new(0.5, 1.0, 0.5),
            Vec3d::new(0.0, -1.0, 0.0),
            0.25,
            0.25,
        );
        assert!(move_result.collided_y);
        assert!((move_result.delta.y + 0.5).abs() < 1.0e-12);
    }

    #[test]
    fn falling_water_ignores_non_sturdy_partial_faces() {
        let mut region = region_with_shape([2, 2, 1]);
        region
            .set_block([0, 0, 0], &parse_block("minecraft:water[level=8]"))
            .unwrap();
        region
            .set_block(
                [1, 0, 0],
                &parse_block("minecraft:oak_fence_gate[facing=north,open=false]"),
            )
            .unwrap();
        let gate_flow = fluid_flow(
            &region,
            [0, 0, 0],
            water_at(&region, [0, 0, 0]).expect("water"),
        );
        assert!(gate_flow.y.abs() < 1.0e-12);

        region
            .set_block([1, 0, 0], &parse_block("minecraft:stone"))
            .unwrap();
        let stone_flow = fluid_flow(
            &region,
            [0, 0, 0],
            water_at(&region, [0, 0, 0]).expect("water"),
        );
        assert!(stone_flow.y < -0.9);
    }

    #[test]
    fn move_entity_matches_vanilla_horizontal_axis_order() {
        let mut region = region_with_shape([3, 1, 3]);
        region
            .set_block([1, 0, 1], &parse_block("minecraft:stone"))
            .unwrap();
        let move_result = move_entity(
            &region,
            Vec3d::new(0.5, 0.0, 0.5),
            Vec3d::new(0.4, 0.0, 0.8),
            0.25,
            0.25,
        );
        assert!((move_result.delta.x - 0.375).abs() < 1.0e-12);
        assert!((move_result.delta.z - 0.8).abs() < 1.0e-12);
        assert!(move_result.collided_x);
        assert!(!move_result.collided_z);
    }

    #[test]
    fn move_towards_closest_space_prefers_open_neighbor() {
        let mut region = region_with_shape([3, 3, 3]);
        for pos in [[1, 1, 1], [0, 1, 1], [2, 1, 1], [1, 1, 2], [1, 2, 1]] {
            region
                .set_block(pos, &parse_block("minecraft:stone"))
                .unwrap();
        }
        let adjusted = move_towards_closest_space(
            &region,
            Vec3d::new(1.8, 1.0, 1.4),
            Vec3d::new(0.4, 0.2, 0.3),
            0.25,
        );
        assert!((adjusted.x - 0.3).abs() < 1.0e-12);
        assert!((adjusted.y - 0.15).abs() < 1.0e-12);
        assert!((adjusted.z + 0.2).abs() < 1.0e-12);
    }

    #[test]
    fn simulate_reports_no_physics_and_pushout() {
        let mut region = region_with_shape([3, 3, 3]);
        for pos in [[1, 1, 1], [0, 1, 1], [2, 1, 1], [1, 1, 2], [1, 2, 1]] {
            region
                .set_block(pos, &parse_block("minecraft:stone"))
                .unwrap();
        }
        let world = LoadedSchematic {
            name: "stuck".to_string(),
            region,
            approximate_collision_blocks: Vec::new(),
        };
        let command = VerifyCommand {
            input: std::path::PathBuf::from("stuck.litematic"),
            out: std::path::PathBuf::from("artifacts/test"),
            target_speed: 0.0,
            ticks: 1,
            inspect_tick: Some(1),
            start_x: 1.8,
            start_y: 1.0,
            start_z: 1.4,
            start_vx: 0.4,
            start_vy: 0.2,
            start_vz: 0.3,
            start_on_ground: false,
            width: VERIFY_DEFAULT_WIDTH,
            height: VERIFY_DEFAULT_HEIGHT,
            entity_id_mod4: 0,
            initial_tick_count: 0,
        };

        let rows = simulate(&world, &command);
        assert!(rows[0].no_physics);
        assert!(rows[1].pushout_applied);
        assert!(rows[1].vz < 0.0);
    }

    #[test]
    fn open_fence_gates_do_not_block_motion() {
        let mut region = region_with_shape([3, 2, 1]);
        region
            .set_block(
                [1, 0, 0],
                &parse_block("minecraft:oak_fence_gate[facing=north,open=true]"),
            )
            .unwrap();
        let move_result = move_entity(
            &region,
            Vec3d::new(0.5, 0.0, 0.5),
            Vec3d::new(1.0, 0.0, 0.0),
            0.25,
            0.25,
        );
        assert!(!move_result.collided_x);
        assert!((move_result.delta.x - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn closed_fence_gates_block_motion() {
        let mut region = region_with_shape([3, 2, 1]);
        region
            .set_block(
                [1, 0, 0],
                &parse_block("minecraft:oak_fence_gate[facing=north,open=false]"),
            )
            .unwrap();
        let move_result = move_entity(
            &region,
            Vec3d::new(0.5, 0.0, 0.5),
            Vec3d::new(1.0, 0.0, 0.0),
            0.25,
            0.25,
        );
        assert!(move_result.collided_x);
        assert!(move_result.delta.x < 1.0);
    }

    #[test]
    fn partial_horizontal_faces_do_not_fully_occlude_flow() {
        let source = parse_block("minecraft:oak_slab[type=top]");
        let target = parse_block("minecraft:oak_slab[type=top]");
        let region = region_with_shape([1, 1, 1]);
        assert!(can_pass_through_wall(
            &region,
            [0, 0, 0],
            &source,
            [1, 0, 0],
            &target,
            HorizontalDir::East,
        ));
    }

    #[test]
    fn spread_to_waterloggable_block_keeps_block_and_sets_waterlogged() {
        let mut region = region_with_shape([1, 1, 1]);
        let slab = parse_block("minecraft:oak_slab[type=bottom]");
        region.set_block([0, 0, 0], &slab).unwrap();
        let mut fluid_ticks = DynamicFluidTicks::default();
        fluid_ticks.spread_to(
            &mut region,
            [0, 0, 0],
            &slab,
            DynamicWaterState {
                amount: 8,
                falling: false,
            },
            0,
        );
        let block = block_at(&region, [0, 0, 0]).expect("waterlogged slab");
        assert_eq!(block.id, "oak_slab");
        assert_eq!(
            block.attributes.get("waterlogged").map(String::as_str),
            Some("true")
        );
    }

    #[test]
    fn spread_to_double_slab_replaces_with_water_block() {
        let mut region = region_with_shape([1, 1, 1]);
        let slab = parse_block("minecraft:oak_slab[type=double]");
        region.set_block([0, 0, 0], &slab).unwrap();
        let mut fluid_ticks = DynamicFluidTicks::default();
        fluid_ticks.spread_to(
            &mut region,
            [0, 0, 0],
            &slab,
            DynamicWaterState {
                amount: 8,
                falling: false,
            },
            0,
        );
        let block = block_at(&region, [0, 0, 0]).expect("water block");
        assert_eq!(block.id, "water");
    }

    #[test]
    fn complex_static_gate_fluid_ticks_match_vanilla_lane40_snapshot() {
        let world = complex_static_gate_world();
        let mut region = world.region.clone();
        run_fluid_ticks(&mut region, 40);

        let expected = [
            (0, "minecraft:water"),
            (1, "minecraft:water[level=1]"),
            (2, "minecraft:water[level=2]"),
            (3, "minecraft:water[level=3]"),
            (4, "minecraft:oak_fence_gate[facing=north,open=true]"),
            (5, "minecraft:water[level=5]"),
            (6, "minecraft:water[level=4]"),
            (7, "minecraft:water[level=3]"),
            (8, "minecraft:water[level=2]"),
            (9, "minecraft:water[level=1]"),
            (10, "minecraft:water"),
        ];

        for (x, expected_block) in expected {
            let block = block_at(&region, [x, 1, 1]).expect("lane block");
            assert_eq!(
                block_full_id(block),
                expected_block,
                "unexpected block at x={x}"
            );
        }
    }

    #[test]
    fn verify_matches_vanilla_complex_static_gate_checkpoints() {
        let world = complex_static_gate_world();
        let command = VerifyCommand {
            input: std::path::PathBuf::from("complex-static-gate.litematic"),
            out: std::path::PathBuf::from("artifacts/test"),
            target_speed: 0.5,
            ticks: 80,
            inspect_tick: Some(80),
            start_x: 1.5,
            start_y: 1.2,
            start_z: 1.5,
            start_vx: 0.0,
            start_vy: 0.0,
            start_vz: 0.0,
            start_on_ground: false,
            width: VERIFY_DEFAULT_WIDTH,
            height: VERIFY_DEFAULT_HEIGHT,
            entity_id_mod4: 0,
            initial_tick_count: 0,
        };

        let rows = simulate(&world, &command);
        let tick1 = &rows[1];
        let tick12 = &rows[12];
        let tick40 = &rows[40];
        let tick80 = &rows[80];
        let epsilon_x = 1.0e-7;
        let epsilon_vx = 1.0e-8;

        assert!((tick1.x - 1.5138600001335143).abs() < 1.0e-12);
        assert!((tick1.y - 1.2005000000237487).abs() < 1.0e-12);
        assert!((tick1.vx - 0.02758280039520264).abs() < 1.0e-12);
        assert!((tick1.vy - 0.0004900000328104948).abs() < 1.0e-12);

        assert!((tick12.x - 3.3010064938397075).abs() < 1.0e-12);
        assert!((tick12.y - 1.2362779907672022).abs() < 1.0e-12);
        assert!((tick12.vx - 0.2817879372128922).abs() < 1.0e-12);
        assert!((tick12.vy - 0.0052744411615881165).abs() < 1.0e-12);

        assert!((tick40.x - 5.667892868262478).abs() < epsilon_x);
        assert!((tick40.y - 1.0845099778976832).abs() < 1.0e-12);
        assert!((tick40.vx + 0.2564466238505471).abs() < epsilon_vx);
        assert!((tick40.vy - 0.007809802505171431).abs() < 1.0e-12);

        assert!((tick80.x - 5.259532379313279).abs() < epsilon_x);
        assert!((tick80.y - 1.0362779907672022).abs() < 1.0e-12);
        assert!((tick80.vx + 0.13993332005607179).abs() < epsilon_vx);
        assert!((tick80.vy - 0.0052744411615881165).abs() < 1.0e-12);
    }

    #[test]
    fn waterlogged_static_lane_matches_vanilla_snapshot() {
        let world = waterlogged_static_world();
        let mut region = world.region.clone();
        run_fluid_ticks(&mut region, 40);

        let source = block_at(&region, [1, 1, 1]).expect("source water");
        assert_eq!(block_full_id(source), "minecraft:water");

        let slab = block_at(&region, [2, 1, 1]).expect("slab");
        assert_eq!(slab.id, "oak_slab");
        assert_ne!(
            slab.attributes.get("waterlogged").map(String::as_str),
            Some("true")
        );

        let air_after_slab = block_at(&region, [3, 1, 1]).expect("air after slab");
        assert!(air_after_slab.is_air());

        let wall = block_at(&region, [4, 1, 1]).expect("wall");
        assert_eq!(wall.id, "cobblestone_wall");
        assert_ne!(
            wall.attributes.get("waterlogged").map(String::as_str),
            Some("true")
        );

        let air_after_wall = block_at(&region, [5, 1, 1]).expect("air after wall");
        assert!(air_after_wall.is_air());

        let bars = block_at(&region, [6, 1, 1]).expect("iron bars");
        assert_eq!(bars.id, "iron_bars");
        assert_ne!(
            bars.attributes.get("waterlogged").map(String::as_str),
            Some("true")
        );

        let air_after_bars = block_at(&region, [7, 1, 1]).expect("air after bars");
        assert!(air_after_bars.is_air());

        let trapdoor = block_at(&region, [8, 1, 1]).expect("trapdoor");
        assert_eq!(trapdoor.id, "oak_trapdoor");
        assert_ne!(
            trapdoor.attributes.get("waterlogged").map(String::as_str),
            Some("true")
        );

        let air_after_trapdoor = block_at(&region, [9, 1, 1]).expect("air after trapdoor");
        assert!(air_after_trapdoor.is_air());

        let stairs = block_at(&region, [10, 1, 1]).expect("stairs");
        assert_eq!(stairs.id, "oak_stairs");
        assert_ne!(
            stairs.attributes.get("waterlogged").map(String::as_str),
            Some("true")
        );

        let trailing_flow = block_at(&region, [11, 1, 1]).expect("trailing flow");
        assert_eq!(block_full_id(trailing_flow), "minecraft:water[level=1]");

        let trailing_source = block_at(&region, [12, 1, 1]).expect("trailing source");
        assert_eq!(block_full_id(trailing_source), "minecraft:water");
    }

    #[test]
    fn verify_matches_vanilla_first_tick_in_simple_channel() {
        let world = simple_channel_world();
        let command = VerifyCommand {
            input: std::path::PathBuf::from("simple-channel.litematic"),
            out: std::path::PathBuf::from("artifacts/test"),
            target_speed: 0.5,
            ticks: 1,
            inspect_tick: Some(1),
            start_x: 2.5,
            start_y: 1.2,
            start_z: 1.5,
            start_vx: 0.0,
            start_vy: 0.0,
            start_vz: 0.0,
            start_on_ground: false,
            width: VERIFY_DEFAULT_WIDTH,
            height: VERIFY_DEFAULT_HEIGHT,
            entity_id_mod4: 0,
            initial_tick_count: 0,
        };

        let rows = simulate(&world, &command);
        let tick = &rows[1];
        assert!((tick.x - 2.5138600001335143).abs() < 1.0e-12);
        assert!((tick.y - 1.2005000000237487).abs() < 1.0e-12);
        assert!((tick.vx - 0.02758280039520264).abs() < 1.0e-12);
        assert!((tick.vy - 0.0004900000328104948).abs() < 1.0e-12);
        assert!(!tick.on_ground);
        assert!(tick.in_water);
        assert!((tick.applied_current_x - 0.028).abs() < 1.0e-12);
    }
}
