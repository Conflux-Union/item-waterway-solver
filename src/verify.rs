use crate::litematic::{
    CollisionBox, FaceDirection, FluidKind, LoadedSchematic, WaterCell, block_at, block_full_id,
    blocks_motion, collision_boxes, fluid_at, is_collision_shape_full_block, is_face_sturdy,
    lava_at, load_litematic, merged_face_occludes, water_at,
};
use crate::{
    AABB_DEFLATE, BUOYANCY, BUOYANCY_CAP, FLUID_CURRENT_EPSILON2, FLUID_CURRENT_MIN_IMPULSE,
    FLUID_CURRENT_MIN_OLD_MOVEMENT, FLUID_MOVEMENT_THRESHOLD, GRAVITY, HORIZONTAL_MOVEMENT_DAMPING,
    HORIZONTAL_REST_THRESHOLD2, HORIZONTAL_WATER_DAMPING, MOVEMENT_SAMPLE_MODULO,
    SLIME_STEP_ON_BASE, SLIME_STEP_ON_VY_SCALE, SLIME_STEP_ON_VY_THRESHOLD, VERIFY_DEFAULT_HEIGHT,
    VERIFY_DEFAULT_WIDTH, VERTICAL_MOVEMENT_DAMPING, VerifyCommand, VerifyEntityKind, WATER_PUSH,
};
use fastnbt::Value;
use mc_schem::region::{BlockEntity, WorldSlice};
use mc_schem::{Block, Region};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use time::OffsetDateTime;
use time::format_description::well_known::Rfc3339;
use uuid::Uuid;

const VERIFY_COLLISION_MODEL: &str = "axis_aligned_block_shapes_with_supported_partials";
const VERIFY_FLUID_MODEL: &str = "guardian_base_tick_plus_post_move_water_current";
const NO_PHYSICS_DEFLATE: f64 = 1.0e-7;
const NO_PHYSICS_PUSHOUT_SPEED: f64 = 0.2;
const NO_PHYSICS_PUSHOUT_SPEED_MIN: f64 = 0.1;
const NO_PHYSICS_PUSHOUT_SPEED_MAX: f64 = 0.3;
const ITEM_MOVEMENT_SUPPORT_OFFSET: f64 = 0.999_999_f32 as f64;
const ENTITY_MOVEMENT_SUPPORT_OFFSET: f64 = 0.500_001_f32 as f64;
const ON_POS_LEGACY_OFFSET: f64 = 0.2_f32 as f64;
const HONEY_SLIDE_TOP_Y: f64 = 0.9375;
const HONEY_SLIDE_MIN_OLD_DELTA_Y: f64 = -0.08;
const HONEY_SLIDE_STRONG_MIN_OLD_DELTA_Y: f64 = -0.13;
const HONEY_SLIDE_TARGET_OLD_DELTA_Y: f64 = -0.05;
const BUBBLE_COLUMN_SURFACE_ACCELERATION: f64 = 0.1;
const BUBBLE_COLUMN_INTERNAL_ACCELERATION: f64 = 0.06;
const BUBBLE_COLUMN_DRAG_DOWN_ACCELERATION: f64 = 0.03;
const BUBBLE_COLUMN_CHECK_TICK_DELAY: usize = 5;
const BUBBLE_COLUMN_FORM_TICK_DELAY: usize = 20;
const BIG_DRIPLEAF_SURVIVAL_TICK_DELAY: usize = 1;
const BIG_DRIPLEAF_UNSTABLE_TICK_DELAY: usize = 10;
const BIG_DRIPLEAF_PARTIAL_TICK_DELAY: usize = 10;
const BIG_DRIPLEAF_FULL_TICK_DELAY: usize = 100;
const POWDER_SNOW_FALL_DISTANCE_COLLISION_THRESHOLD: f64 = 2.5;
const FLOWING_WATER_TICK_DELAY: usize = 5;
const SCAFFOLDING_TICK_DELAY: usize = 1;
const BAMBOO_TICK_DELAY: usize = 1;
const CACTUS_TICK_DELAY: usize = 1;
const LADDER_TICK_DELAY: usize = 1;
const GROUND_SUPPORT_PROBE_DELTA: f64 = 1.0e-5;
const WATER_SLOPE_FIND_DISTANCE: usize = 4;
const LAVA_FLOW_SCALE: f64 = 0.002_333_333_333_333_333_5;
const HORIZONTAL_LAVA_DAMPING: f64 = 0.95;
const ENTITY_BASE_GRAVITY: f64 = 0.08;
const LIVING_AIR_DRAG: f64 = 0.91_f32 as f64;
const LIVING_VERTICAL_AIR_DRAG: f64 = 0.98_f32 as f64;
const LIVING_WATER_DRAG: f64 = 0.8_f32 as f64;
const LIVING_LAVA_DRAG: f64 = 0.5_f32 as f64;
const LIVING_LAVA_VERTICAL_DRAG: f64 = 0.8_f32 as f64;
const LIVING_CLIMBABLE_MAX_DELTA: f64 = 0.15_f32 as f64;
const LIVING_CLIMBABLE_ASCENT: f64 = 0.2;
const LIVING_FLUID_JUMP_OUT_CLEARANCE: f64 = 0.6_f32 as f64;
const LIVING_FLUID_JUMP_OUT_VELOCITY: f64 = 0.3_f32 as f64;

const LEGACY_RANDOM_MULTIPLIER: u64 = 25_214_903_917;
const LEGACY_RANDOM_INCREMENT: u64 = 11;
const LEGACY_RANDOM_MASK: u64 = (1_u64 << 48) - 1;

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

fn uses_item_movement_sampling(entity_kind: VerifyEntityKind) -> bool {
    matches!(entity_kind, VerifyEntityKind::Item)
}

fn hopper_collects_entity(entity_kind: VerifyEntityKind) -> bool {
    matches!(entity_kind, VerifyEntityKind::Item)
}

fn has_post_move_fluid_current(entity_kind: VerifyEntityKind) -> bool {
    matches!(entity_kind, VerifyEntityKind::Item)
}

fn skips_active_travel(entity_kind: VerifyEntityKind, no_ai: bool) -> bool {
    matches!(entity_kind, VerifyEntityKind::Living) && no_ai
}

fn apply_living_delta_deadzone(entity_kind: VerifyEntityKind, vel: &mut Vec3d) {
    if !matches!(entity_kind, VerifyEntityKind::Living) {
        return;
    }
    if vel.x.abs() < 0.003 {
        vel.x = 0.0;
    }
    if vel.y.abs() < 0.003 {
        vel.y = 0.0;
    }
    if vel.z.abs() < 0.003 {
        vel.z = 0.0;
    }
}

fn powder_snow_has_walkable_collision(entity_kind: VerifyEntityKind) -> bool {
    matches!(entity_kind, VerifyEntityKind::FallingBlock)
}

fn powder_snow_inside_effect_applies(
    entity_kind: VerifyEntityKind,
    in_block_powder_snow: bool,
) -> bool {
    !matches!(entity_kind, VerifyEntityKind::Living) || in_block_powder_snow
}

fn default_item_health(entity_kind: VerifyEntityKind) -> Option<i32> {
    matches!(entity_kind, VerifyEntityKind::Item).then_some(5)
}

fn base_gravity(entity_kind: VerifyEntityKind, no_gravity: bool) -> f64 {
    if no_gravity {
        return 0.0;
    }
    match entity_kind {
        VerifyEntityKind::Item | VerifyEntityKind::FallingBlock => GRAVITY,
        VerifyEntityKind::Generic | VerifyEntityKind::Living => ENTITY_BASE_GRAVITY,
    }
}

fn movement_support_offset(entity_kind: VerifyEntityKind) -> f64 {
    match entity_kind {
        VerifyEntityKind::Item => ITEM_MOVEMENT_SUPPORT_OFFSET,
        VerifyEntityKind::Generic | VerifyEntityKind::Living | VerifyEntityKind::FallingBlock => {
            ENTITY_MOVEMENT_SUPPORT_OFFSET
        }
    }
}

fn fluid_jump_threshold(entity_height: f64) -> f64 {
    if entity_height < 0.4 { 0.0 } else { 0.4 }
}

fn living_fluid_falling_adjusted_movement(
    base_gravity: f64,
    is_falling: bool,
    movement: Vec3d,
) -> Vec3d {
    if base_gravity == 0.0 {
        return movement;
    }
    let yd = if is_falling
        && (movement.y - 0.005).abs() >= 0.003
        && (movement.y - base_gravity / 16.0).abs() < 0.003
    {
        -0.003
    } else {
        movement.y - base_gravity / 16.0
    };
    Vec3d::new(movement.x, yd, movement.z)
}

fn apply_living_air_physics(
    vel: &mut Vec3d,
    on_ground: bool,
    profile: GroundProfile,
    gravity: f64,
) {
    let block_friction = if on_ground { profile.friction } else { 1.0 };
    let friction = block_friction * LIVING_AIR_DRAG;
    vel.y -= gravity;
    vel.x *= friction;
    vel.y *= LIVING_VERTICAL_AIR_DRAG;
    vel.z *= friction;
}

fn apply_living_water_physics(vel: &mut Vec3d, gravity: f64, is_falling: bool) {
    let movement = Vec3d::new(
        vel.x * LIVING_WATER_DRAG,
        vel.y * LIVING_WATER_DRAG,
        vel.z * LIVING_WATER_DRAG,
    );
    *vel = living_fluid_falling_adjusted_movement(gravity, is_falling, movement);
}

fn apply_living_lava_physics(
    vel: &mut Vec3d,
    gravity: f64,
    is_falling: bool,
    fluid_height: f64,
    entity_height: f64,
) {
    if fluid_height <= fluid_jump_threshold(entity_height) {
        let movement = Vec3d::new(
            vel.x * LIVING_LAVA_DRAG,
            vel.y * LIVING_LAVA_VERTICAL_DRAG,
            vel.z * LIVING_LAVA_DRAG,
        );
        *vel = living_fluid_falling_adjusted_movement(gravity, is_falling, movement);
    } else {
        vel.x *= LIVING_LAVA_DRAG;
        vel.y *= LIVING_LAVA_DRAG;
        vel.z *= LIVING_LAVA_DRAG;
    }
    if gravity != 0.0 {
        vel.y -= gravity / 4.0;
    }
}

fn clear_fire(remaining_fire_ticks: &mut i32) {
    *remaining_fire_ticks = (*remaining_fire_ticks).min(0);
}

fn ignite_for_ticks(remaining_fire_ticks: &mut i32, number_of_ticks: i32) {
    if *remaining_fire_ticks < number_of_ticks {
        *remaining_fire_ticks = number_of_ticks;
    }
}

fn fire_ignite(remaining_fire_ticks: &mut i32, fire_immune: bool) {
    if fire_immune {
        return;
    }
    if *remaining_fire_ticks < 0 {
        *remaining_fire_ticks += 1;
    }
    if *remaining_fire_ticks >= 0 {
        ignite_for_ticks(remaining_fire_ticks, 8 * 20);
    }
}

fn lava_ignite(remaining_fire_ticks: &mut i32, fire_immune: bool) {
    if !fire_immune {
        ignite_for_ticks(remaining_fire_ticks, 15 * 20);
    }
}

fn damage_tracked_entity(
    health: &mut Option<i32>,
    amount: i32,
    reason: &'static str,
    alive: &mut bool,
    removed_by: &mut Option<&'static str>,
) {
    let Some(health) = health.as_mut() else {
        return;
    };
    *health -= amount;
    if *health <= 0 {
        *alive = false;
        *removed_by = Some(reason);
    }
}

fn lava_hurt(
    fire_immune: bool,
    item_health: &mut Option<i32>,
    alive: &mut bool,
    removed_by: &mut Option<&'static str>,
) {
    if fire_immune {
        return;
    }
    damage_tracked_entity(item_health, 4, "lavaHurt", alive, removed_by);
}

fn on_fire_hurt(
    fire_immune: bool,
    item_health: &mut Option<i32>,
    alive: &mut bool,
    removed_by: &mut Option<&'static str>,
) {
    on_fire_hurt_with_damage(fire_immune, item_health, 1, alive, removed_by);
}

fn on_fire_hurt_with_damage(
    fire_immune: bool,
    item_health: &mut Option<i32>,
    damage: i32,
    alive: &mut bool,
    removed_by: &mut Option<&'static str>,
) {
    if fire_immune {
        return;
    }
    damage_tracked_entity(item_health, damage, "onFireDamage", alive, removed_by);
}

fn tick_fire_state(
    is_in_lava: bool,
    fire_immune: bool,
    remaining_fire_ticks: &mut i32,
    item_health: &mut Option<i32>,
    alive: &mut bool,
    removed_by: &mut Option<&'static str>,
) {
    if *remaining_fire_ticks <= 0 {
        return;
    }
    if fire_immune {
        clear_fire(remaining_fire_ticks);
        return;
    }
    if *remaining_fire_ticks % 20 == 0 && !is_in_lava {
        on_fire_hurt(fire_immune, item_health, alive, removed_by);
    }
    *remaining_fire_ticks -= 1;
}

#[derive(Clone, Copy, Debug)]
struct GroundProfile {
    friction: f64,
    slime_surface: bool,
}

#[derive(Clone, Copy, Debug, Default)]
struct FluidTracker {
    height: f64,
    accumulated_current: Vec3d,
    current_count: usize,
}

impl FluidTracker {
    fn is_in_fluid(self) -> bool {
        self.height > 0.0
    }

    fn applies_movement_damping(self) -> bool {
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

#[derive(Clone, Copy, Debug)]
struct LegacyRandom {
    seed: u64,
}

impl LegacyRandom {
    fn new(seed: i64) -> Self {
        Self {
            seed: ((seed as u64) ^ LEGACY_RANDOM_MULTIPLIER) & LEGACY_RANDOM_MASK,
        }
    }

    fn from_internal_seed(seed: u64) -> Self {
        Self {
            seed: seed & LEGACY_RANDOM_MASK,
        }
    }

    fn next(&mut self, bits: u32) -> u32 {
        self.seed = (self
            .seed
            .wrapping_mul(LEGACY_RANDOM_MULTIPLIER)
            .wrapping_add(LEGACY_RANDOM_INCREMENT))
            & LEGACY_RANDOM_MASK;
        (self.seed >> (48 - bits)) as u32
    }

    fn next_float(&mut self) -> f64 {
        self.next(24) as f64 / (1_u32 << 24) as f64
    }

    #[cfg(test)]
    fn next_long(&mut self) -> u64 {
        ((self.next(32) as u64) << 32) | self.next(32) as u64
    }
}

fn next_legacy_seed(seed: u64) -> u64 {
    seed.wrapping_mul(LEGACY_RANDOM_MULTIPLIER)
        .wrapping_add(LEGACY_RANDOM_INCREMENT)
        & LEGACY_RANDOM_MASK
}

fn legacy_random_from_entity_uuid(uuid_text: &str) -> Option<LegacyRandom> {
    let uuid = Uuid::parse_str(uuid_text).ok()?;
    let seed = legacy_random_state_after_entity_uuid(uuid)?;
    Some(LegacyRandom::from_internal_seed(seed))
}

fn legacy_random_state_after_entity_uuid(uuid: Uuid) -> Option<u64> {
    let bytes = *uuid.as_bytes();
    let most = u64::from_be_bytes(bytes[..8].try_into().ok()?);
    let least = u64::from_be_bytes(bytes[8..].try_into().ok()?);
    let output1 = (most >> 32) as u32;
    let output2_masked = most as u32;
    let output3_masked = (least >> 32) as u32;
    let output4 = least as u32;

    for version_bits in 0_u32..16 {
        let output2 = (output2_masked & !0x0000_F000) | (version_bits << 12);
        for variant_bits in 0_u32..4 {
            let output3 = (output3_masked & 0x3fff_ffff) | (variant_bits << 30);
            if let Some(seed) = recover_legacy_seed_after_uuid(output1, output2, output3, output4) {
                return Some(seed);
            }
        }
    }

    None
}

fn recover_legacy_seed_after_uuid(
    output1: u32,
    output2: u32,
    output3: u32,
    output4: u32,
) -> Option<u64> {
    for low16 in 0_u64..=0xffff {
        let seed1 = ((output1 as u64) << 16) | low16;
        let seed2 = next_legacy_seed(seed1);
        if (seed2 >> 16) as u32 != output2 {
            continue;
        }
        let seed3 = next_legacy_seed(seed2);
        if (seed3 >> 16) as u32 != output3 {
            continue;
        }
        let seed4 = next_legacy_seed(seed3);
        if (seed4 >> 16) as u32 == output4 {
            return Some(seed4);
        }
    }
    None
}

#[cfg(test)]
fn insecure_uuid_from_legacy_random(random: &mut LegacyRandom) -> Uuid {
    let most = (random.next_long() & !0x0000_0000_0000_f000) | 0x0000_0000_0000_4000;
    let least = (random.next_long() & 0x3fff_ffff_ffff_ffff) | 0x8000_0000_0000_0000;
    Uuid::from_u64_pair(most, least)
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
                    let Some(_fluid) = dynamic_water_state_at(region, pos) else {
                        continue;
                    };
                    let Some(_block_state) = block_at(region, pos) else {
                        continue;
                    };
                    ticks.schedule(FLOWING_WATER_TICK_DELAY, pos);
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

    fn run_due(&mut self, region: &mut Region, block_ticks: &mut DynamicBlockTicks, tick: usize) {
        let Some(due) = self.scheduled.remove(&tick) else {
            return;
        };
        for pos in due {
            self.tick_water(region, block_ticks, pos, tick);
        }
    }

    fn tick_water(
        &mut self,
        region: &mut Region,
        block_ticks: &mut DynamicBlockTicks,
        pos: [i32; 3],
        tick: usize,
    ) {
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
                    block_ticks.schedule_bubble_updates_around(region, tick, pos);
                    block_state = block_at(region, pos).cloned().expect("water block exists");
                }
                None => {
                    set_air_block(region, pos);
                    self.schedule_neighbors(tick + FLOWING_WATER_TICK_DELAY, pos);
                    block_ticks.schedule_bubble_updates_around(region, tick, pos);
                    return;
                }
                _ => {}
            }
        }

        self.spread(region, block_ticks, pos, &block_state, fluid, tick);
    }

    fn spread(
        &mut self,
        region: &mut Region,
        block_ticks: &mut DynamicBlockTicks,
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
                    if below_fluid.is_none()
                        && can_hold_specific_fluid(&below_block, new_below_fluid)
                    {
                        self.spread_to(
                            region,
                            block_ticks,
                            below_pos,
                            &below_block,
                            new_below_fluid,
                            tick,
                        );
                        if source_neighbor_count(region, pos) >= 3 {
                            self.spread_to_sides(
                                region,
                                block_ticks,
                                pos,
                                fluid,
                                block_state,
                                tick,
                            );
                        }
                        return;
                    }
                }
            }

            if fluid.is_source()
                || !is_water_hole(region, pos, block_state, below_pos, &below_block)
            {
                self.spread_to_sides(region, block_ticks, pos, fluid, block_state, tick);
            }
        }
    }

    fn spread_to_sides(
        &mut self,
        region: &mut Region,
        block_ticks: &mut DynamicBlockTicks,
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
                self.spread_to(
                    region,
                    block_ticks,
                    neighbor_pos,
                    &neighbor_block,
                    new_fluid,
                    tick,
                );
            }
        }
    }

    fn spread_to(
        &mut self,
        region: &mut Region,
        block_ticks: &mut DynamicBlockTicks,
        pos: [i32; 3],
        state: &Block,
        fluid: DynamicWaterState,
        tick: usize,
    ) {
        if is_waterloggable_block(state) {
            if try_place_waterlogged_block(region, pos, state, fluid) {
                self.schedule_neighbors(tick + FLOWING_WATER_TICK_DELAY, pos);
                block_ticks.schedule_bubble_updates_around(region, tick, pos);
            }
        } else {
            set_water_block(region, pos, fluid);
            self.schedule_neighbors(tick + FLOWING_WATER_TICK_DELAY, pos);
            block_ticks.schedule_bubble_updates_around(region, tick, pos);
        }
    }
}

#[derive(Default)]
struct DynamicBlockTicks {
    scheduled: BTreeMap<usize, BTreeSet<[i32; 3]>>,
    big_dripleaf_tilts: BTreeMap<usize, BTreeSet<[i32; 3]>>,
}

impl DynamicBlockTicks {
    fn bootstrap(region: &Region) -> Self {
        let mut ticks = Self::default();
        let shape = region.shape();
        for x in 0..shape[0] {
            for y in 0..shape[1] {
                for z in 0..shape[2] {
                    let pos = [x, y, z];
                    let Some(block) = block_at(region, pos) else {
                        continue;
                    };
                    if block.namespace == "minecraft" && block.id == "scaffolding" {
                        ticks.schedule(SCAFFOLDING_TICK_DELAY, pos);
                    }
                    ticks.schedule_pointed_dripstone_tick_if_needed(region, 0, pos);
                    ticks.schedule_big_dripleaf_tick_if_needed(region, 0, pos);
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
            if let Some(tilt_due) = self.big_dripleaf_tilts.remove(&tick) {
                for pos in tilt_due {
                    self.tick_big_dripleaf(region, pos, tick);
                }
            }
            return;
        };
        for pos in due {
            self.tick_bamboo(region, pos, tick);
            self.tick_cactus(region, pos, tick);
            self.tick_ladder(region, pos, tick);
            self.tick_scaffolding(region, pos, tick);
            self.tick_pointed_dripstone(region, pos, tick);
            self.tick_big_dripleaf_survival(region, pos, tick);
            self.tick_big_dripleaf_stem(region, pos, tick);
            self.tick_bubble_column(region, pos);
        }
        if let Some(tilt_due) = self.big_dripleaf_tilts.remove(&tick) {
            for pos in tilt_due {
                self.tick_big_dripleaf(region, pos, tick);
            }
        }
    }

    fn tick_bamboo(&mut self, region: &mut Region, pos: [i32; 3], tick: usize) {
        let Some(block) = block_at(region, pos).cloned() else {
            return;
        };
        if block.namespace != "minecraft" {
            return;
        }
        match block.id.as_str() {
            "bamboo" => {
                if bamboo_can_survive(region, pos) {
                    return;
                }
                set_air_block(region, pos);
                self.schedule_neighbors(tick + BAMBOO_TICK_DELAY, pos);
            }
            "bamboo_sapling" => {
                if !bamboo_can_survive(region, pos) {
                    set_air_block(region, pos);
                    self.schedule_neighbors(tick + BAMBOO_TICK_DELAY, pos);
                }
            }
            _ => {}
        }
    }

    fn tick_cactus(&mut self, region: &mut Region, pos: [i32; 3], tick: usize) {
        let Some(block) = block_at(region, pos).cloned() else {
            return;
        };
        if block.namespace != "minecraft" || block.id != "cactus" {
            return;
        }
        if cactus_can_survive(region, pos) {
            return;
        }
        set_air_block(region, pos);
        self.schedule_neighbors(tick + CACTUS_TICK_DELAY, pos);
    }

    fn tick_ladder(&mut self, region: &mut Region, pos: [i32; 3], tick: usize) {
        let Some(block) = block_at(region, pos).cloned() else {
            return;
        };
        if block.namespace != "minecraft" || block.id != "ladder" {
            return;
        }
        if ladder_can_survive(region, pos, &block) {
            return;
        }
        set_air_block(region, pos);
        self.schedule_neighbors(tick + LADDER_TICK_DELAY, pos);
    }

    fn tick_scaffolding(&mut self, region: &mut Region, pos: [i32; 3], tick: usize) {
        let Some(block) = block_at(region, pos).cloned() else {
            return;
        };
        if block.namespace != "minecraft" || block.id != "scaffolding" {
            return;
        }

        let distance = scaffolding_distance(region, pos);
        let bottom = scaffolding_is_bottom(region, pos, distance);
        let current_distance = scaffolding_distance_attr(&block);
        let current_bottom = bool_attr(&block, "bottom");
        if distance == 7 {
            set_air_block(region, pos);
            self.schedule_neighbors(tick + SCAFFOLDING_TICK_DELAY, pos);
            return;
        }

        if current_distance != distance || current_bottom != bottom {
            let updated = with_attr(
                &with_attr(&block, "distance", &distance.to_string()),
                "bottom",
                if bottom { "true" } else { "false" },
            );
            let _ = region.set_block(pos, &updated);
            self.schedule_neighbors(tick + SCAFFOLDING_TICK_DELAY, pos);
        }
    }

    fn tick_bubble_column(&mut self, region: &mut Region, pos: [i32; 3]) {
        if maybe_update_bubble_column(region, pos) {
            self.schedule_bubble_updates_around(region, 0, pos);
        }
    }

    fn tick_pointed_dripstone(&mut self, region: &mut Region, pos: [i32; 3], tick: usize) {
        let Some(block) = block_at(region, pos).cloned() else {
            return;
        };
        if block.namespace != "minecraft" || block.id != "pointed_dripstone" {
            return;
        }
        if pointed_dripstone_can_survive(region, pos, &block) {
            return;
        }
        set_air_block(region, pos);
        self.schedule_neighbors(tick + 1, pos);
    }

    fn tick_big_dripleaf_survival(&mut self, region: &mut Region, pos: [i32; 3], tick: usize) {
        let Some(block) = block_at(region, pos).cloned() else {
            return;
        };
        if block.namespace != "minecraft" || block.id != "big_dripleaf" {
            return;
        }
        if !big_dripleaf_can_survive(region, pos) {
            set_air_block(region, pos);
            self.schedule_neighbors(tick + BIG_DRIPLEAF_SURVIVAL_TICK_DELAY, pos);
            return;
        }
        if block_at(region, offset_pos(pos, [0, 1, 0]))
            .map(|above| above.namespace == "minecraft" && above.id == "big_dripleaf")
            .unwrap_or(false)
        {
            let updated = big_dripleaf_stem_from_leaf(&block);
            let _ = region.set_block(pos, &updated);
            self.schedule_neighbors(tick + BIG_DRIPLEAF_SURVIVAL_TICK_DELAY, pos);
            return;
        }
    }

    fn tick_big_dripleaf(&mut self, region: &mut Region, pos: [i32; 3], tick: usize) {
        let Some(block) = block_at(region, pos).cloned() else {
            return;
        };
        if block.namespace != "minecraft" || block.id != "big_dripleaf" {
            return;
        }
        let tilt = big_dripleaf_tilt(&block);
        if block_has_neighbor_signal(region, pos) {
            if tilt != "none" {
                let _ = region.set_block(pos, &with_attr(&block, "tilt", "none"));
            }
            return;
        }

        let next_tilt = match tilt {
            "unstable" => Some("partial"),
            "partial" => Some("full"),
            "full" => Some("none"),
            _ => None,
        };
        let Some(next_tilt) = next_tilt else {
            return;
        };
        let updated = with_attr(&block, "tilt", next_tilt);
        let _ = region.set_block(pos, &updated);
        self.schedule_big_dripleaf_tick_if_needed(region, tick, pos);
    }

    fn tick_big_dripleaf_stem(&mut self, region: &mut Region, pos: [i32; 3], tick: usize) {
        let Some(block) = block_at(region, pos).cloned() else {
            return;
        };
        if block.namespace != "minecraft" || block.id != "big_dripleaf_stem" {
            return;
        }
        if big_dripleaf_stem_can_survive(region, pos) {
            return;
        }
        set_air_block(region, pos);
        self.schedule_neighbors(tick + BIG_DRIPLEAF_SURVIVAL_TICK_DELAY, pos);
    }

    fn schedule_bubble_updates_around(&mut self, region: &Region, tick: usize, pos: [i32; 3]) {
        for offset in [[0, -1, 0], [0, 0, 0], [0, 1, 0]] {
            self.schedule_bubble_tick_if_needed(region, tick, offset_pos(pos, offset));
        }
    }

    fn schedule_big_dripleaf_tick_if_needed(
        &mut self,
        region: &Region,
        tick: usize,
        pos: [i32; 3],
    ) {
        let Some(block) = block_at(region, pos) else {
            return;
        };
        if block.namespace != "minecraft" || block.id != "big_dripleaf" {
            return;
        }
        if let Some(delay) = big_dripleaf_tick_delay(big_dripleaf_tilt(block)) {
            self.big_dripleaf_tilts
                .entry(tick + delay)
                .or_default()
                .insert(pos);
        }
    }

    fn schedule_pointed_dripstone_tick_if_needed(
        &mut self,
        region: &Region,
        tick: usize,
        pos: [i32; 3],
    ) {
        let Some(block) = block_at(region, pos) else {
            return;
        };
        if block.namespace != "minecraft" || block.id != "pointed_dripstone" {
            return;
        }
        if pointed_dripstone_can_survive(region, pos, block) {
            return;
        }
        let delay = if pointed_dripstone_tip_direction(block) == FaceDirection::Down {
            2
        } else {
            1
        };
        self.schedule(tick + delay, pos);
    }

    fn schedule_bubble_tick_if_needed(&mut self, region: &Region, tick: usize, pos: [i32; 3]) {
        let Some(block) = block_at(region, pos) else {
            return;
        };
        let delay = if block.namespace == "minecraft" && block.id == "bubble_column" {
            Some(BUBBLE_COLUMN_CHECK_TICK_DELAY)
        } else if can_bubble_column_occupy(block)
            && block_at(region, offset_pos(pos, [0, -1, 0]))
                .map(bubble_column_supports_or_extends)
                .unwrap_or(false)
        {
            Some(BUBBLE_COLUMN_FORM_TICK_DELAY)
        } else {
            None
        };
        if let Some(delay) = delay {
            self.schedule(tick + delay, pos);
        }
    }
}

fn big_dripleaf_tilt(block: &Block) -> &str {
    block
        .attributes
        .get("tilt")
        .map(String::as_str)
        .unwrap_or("none")
}

fn pointed_dripstone_tip_direction(block: &Block) -> FaceDirection {
    match block
        .attributes
        .get("vertical_direction")
        .map(String::as_str)
        .unwrap_or("up")
    {
        "down" => FaceDirection::Down,
        _ => FaceDirection::Up,
    }
}

fn pointed_dripstone_can_survive(region: &Region, pos: [i32; 3], block: &Block) -> bool {
    let tip_direction = pointed_dripstone_tip_direction(block);
    let behind_pos = match tip_direction {
        FaceDirection::Up => offset_pos(pos, [0, -1, 0]),
        FaceDirection::Down => offset_pos(pos, [0, 1, 0]),
        _ => return false,
    };
    let Some(behind_block) = block_at(region, behind_pos) else {
        return false;
    };
    is_face_sturdy(behind_block, tip_direction)
        || (behind_block.namespace == "minecraft"
            && behind_block.id == "pointed_dripstone"
            && pointed_dripstone_tip_direction(behind_block) == tip_direction)
}

fn bamboo_can_survive(region: &Region, pos: [i32; 3]) -> bool {
    block_at(region, offset_pos(pos, [0, -1, 0]))
        .map(bamboo_supports_growth)
        .unwrap_or(false)
}

fn bamboo_supports_growth(block: &Block) -> bool {
    block.namespace == "minecraft"
        && matches!(
            block.id.as_str(),
            "bamboo"
                | "bamboo_sapling"
                | "sand"
                | "red_sand"
                | "suspicious_sand"
                | "dirt"
                | "coarse_dirt"
                | "rooted_dirt"
                | "mud"
                | "muddy_mangrove_roots"
                | "moss_block"
                | "pale_moss_block"
                | "grass_block"
                | "podzol"
                | "mycelium"
                | "gravel"
                | "suspicious_gravel"
        )
}

fn big_dripleaf_can_survive(region: &Region, pos: [i32; 3]) -> bool {
    block_at(region, offset_pos(pos, [0, -1, 0]))
        .map(big_dripleaf_supports_or_extends)
        .unwrap_or(false)
}

fn big_dripleaf_stem_can_survive(region: &Region, pos: [i32; 3]) -> bool {
    let below_supported = block_at(region, offset_pos(pos, [0, -1, 0]))
        .map(big_dripleaf_stem_supports)
        .unwrap_or(false);
    let above_supported = block_at(region, offset_pos(pos, [0, 1, 0]))
        .map(big_dripleaf_head_or_stem)
        .unwrap_or(false);
    below_supported && above_supported
}

fn big_dripleaf_supports_or_extends(block: &Block) -> bool {
    block.namespace == "minecraft"
        && matches!(block.id.as_str(), "big_dripleaf" | "big_dripleaf_stem")
        || big_dripleaf_floor_support(block)
}

fn big_dripleaf_stem_supports(block: &Block) -> bool {
    (block.namespace == "minecraft" && block.id == "big_dripleaf_stem")
        || big_dripleaf_floor_support(block)
}

fn big_dripleaf_head_or_stem(block: &Block) -> bool {
    block.namespace == "minecraft"
        && matches!(block.id.as_str(), "big_dripleaf" | "big_dripleaf_stem")
}

fn big_dripleaf_floor_support(block: &Block) -> bool {
    block.namespace == "minecraft"
        && matches!(
            block.id.as_str(),
            "clay"
                | "moss_block"
                | "dirt"
                | "grass_block"
                | "podzol"
                | "coarse_dirt"
                | "mycelium"
                | "rooted_dirt"
                | "mud"
                | "muddy_mangrove_roots"
                | "farmland"
        )
}

fn big_dripleaf_stem_from_leaf(block: &Block) -> Block {
    let facing = block
        .attributes
        .get("facing")
        .map(String::as_str)
        .unwrap_or("north");
    let waterlogged = block
        .attributes
        .get("waterlogged")
        .map(String::as_str)
        .unwrap_or("false");
    let stem = Block::from_id("minecraft:big_dripleaf_stem").expect("valid big dripleaf stem");
    with_attr(
        &with_attr(&stem, "facing", facing),
        "waterlogged",
        waterlogged,
    )
}

fn cactus_can_survive(region: &Region, pos: [i32; 3]) -> bool {
    for direction in HorizontalDir::ALL {
        let neighbor_pos = offset_pos(pos, direction.offset());
        if lava_at(region, neighbor_pos).is_some() {
            return false;
        }
        if block_at(region, neighbor_pos)
            .map(cactus_horizontal_neighbor_is_solid)
            .unwrap_or(false)
        {
            return false;
        }
    }

    let below_supported = block_at(region, offset_pos(pos, [0, -1, 0]))
        .map(cactus_supports_growth)
        .unwrap_or(false);
    if !below_supported {
        return false;
    }

    fluid_at(region, offset_pos(pos, [0, 1, 0]), FluidKind::Water).is_none()
        && fluid_at(region, offset_pos(pos, [0, 1, 0]), FluidKind::Lava).is_none()
}

fn cactus_horizontal_neighbor_is_solid(block: &Block) -> bool {
    if block.is_air() {
        return false;
    }
    if block.namespace != "minecraft" {
        return true;
    }
    is_collision_shape_full_block(block)
}

fn cactus_supports_growth(block: &Block) -> bool {
    block.namespace == "minecraft"
        && matches!(
            block.id.as_str(),
            "cactus" | "sand" | "red_sand" | "suspicious_sand"
        )
}

fn ladder_can_survive(region: &Region, pos: [i32; 3], block: &Block) -> bool {
    let Some(facing) = facing_attr(block).and_then(horizontal_dir_attr) else {
        return false;
    };
    let support_pos = offset_pos(pos, facing.opposite().offset());
    let Some(support_block) = block_at(region, support_pos) else {
        return false;
    };
    is_face_sturdy(support_block, facing.face())
}

fn big_dripleaf_tick_delay(tilt: &str) -> Option<usize> {
    match tilt {
        "unstable" => Some(BIG_DRIPLEAF_UNSTABLE_TICK_DELAY),
        "partial" => Some(BIG_DRIPLEAF_PARTIAL_TICK_DELAY),
        "full" => Some(BIG_DRIPLEAF_FULL_TICK_DELAY),
        _ => None,
    }
}

fn dynamic_water_state_at(region: &Region, pos: [i32; 3]) -> Option<DynamicWaterState> {
    water_at(region, pos).map(DynamicWaterState::from_cell)
}

fn maybe_update_bubble_column(region: &mut Region, pos: [i32; 3]) -> bool {
    let Some(occupy_state) = block_at(region, pos).cloned() else {
        return false;
    };
    if !can_bubble_column_occupy(&occupy_state) {
        return false;
    }

    let Some(updated_state) = bubble_column_state_from_below(region, pos, &occupy_state) else {
        return false;
    };
    if updated_state.full_id() == occupy_state.full_id() {
        return false;
    }

    let mut current_pos = pos;
    loop {
        let Some(current_state) = block_at(region, current_pos).cloned() else {
            break;
        };
        if !can_bubble_column_occupy(&current_state) {
            break;
        }
        let _ = region.set_block(current_pos, &updated_state);
        current_pos = offset_pos(current_pos, [0, 1, 0]);
    }
    true
}

fn can_bubble_column_occupy(block: &Block) -> bool {
    if block.namespace != "minecraft" {
        return false;
    }
    if block.id == "bubble_column" {
        return true;
    }
    block.id == "water"
        && dynamic_water_state_at_block(block)
            .map(DynamicWaterState::is_source)
            .unwrap_or(false)
}

fn bubble_column_state_from_below(
    region: &Region,
    pos: [i32; 3],
    occupy_state: &Block,
) -> Option<Block> {
    let below_state = block_at(region, offset_pos(pos, [0, -1, 0]))?;
    if below_state.namespace == "minecraft" && below_state.id == "bubble_column" {
        return Some(below_state.clone());
    }
    if bubble_column_pushes_up(below_state) {
        return Some(bubble_column_block(false));
    }
    if bubble_column_drags_down(below_state) {
        return Some(bubble_column_block(true));
    }
    if occupy_state.namespace == "minecraft" && occupy_state.id == "bubble_column" {
        return Some(Block::from_id("minecraft:water").expect("valid water source"));
    }
    Some(occupy_state.clone())
}

fn bubble_column_pushes_up(block: &Block) -> bool {
    block.namespace == "minecraft" && block.id == "soul_sand"
}

fn bubble_column_drags_down(block: &Block) -> bool {
    block.namespace == "minecraft" && block.id == "magma_block"
}

fn bubble_column_supports_or_extends(block: &Block) -> bool {
    (block.namespace == "minecraft" && block.id == "bubble_column")
        || bubble_column_pushes_up(block)
        || bubble_column_drags_down(block)
}

fn bubble_column_block(drag_down: bool) -> Block {
    Block::from_id(if drag_down {
        "minecraft:bubble_column[drag=true]"
    } else {
        "minecraft:bubble_column[drag=false]"
    })
    .expect("valid bubble column block")
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
        if !can_hold_specific_fluid(test_state, new_fluid) || test_fluid.is_some() {
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
    dynamic_water_state_at(region, bottom_pos).is_some()
        || can_hold_specific_fluid(
            bottom_state,
            DynamicWaterState {
                amount: 7,
                falling: false,
            },
        )
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

    if block.namespace == "minecraft" && block.id == "bubble_column" {
        return Some(WaterCell {
            height: 1.0,
            own_height: (8.0_f32 / 9.0_f32) as f64,
            falling: false,
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

fn bool_attr(block: &Block, key: &str) -> bool {
    block.attributes.get(key).map(String::as_str) == Some("true")
}

fn facing_attr<'a>(block: &'a Block) -> Option<&'a str> {
    block.attributes.get("facing").map(String::as_str)
}

fn horizontal_dir_attr(value: &str) -> Option<HorizontalDir> {
    match value {
        "north" => Some(HorizontalDir::North),
        "south" => Some(HorizontalDir::South),
        "west" => Some(HorizontalDir::West),
        "east" => Some(HorizontalDir::East),
        _ => None,
    }
}

fn with_attr(block: &Block, key: &str, value: &str) -> Block {
    let mut updated = block.clone();
    updated
        .attributes
        .insert(key.to_string(), value.to_string());
    updated
}

fn block_has_neighbor_signal(region: &Region, pos: [i32; 3]) -> bool {
    for offset in [
        [0, -1, 0],
        [0, 1, 0],
        [-1, 0, 0],
        [1, 0, 0],
        [0, 0, -1],
        [0, 0, 1],
    ] {
        let Some(block) = block_at(region, offset_pos(pos, offset)) else {
            continue;
        };
        if block_emits_neighbor_signal(block) {
            return true;
        }
    }
    false
}

fn block_emits_neighbor_signal(block: &Block) -> bool {
    if block.namespace != "minecraft" {
        return false;
    }

    if block.id == "redstone_block" {
        return true;
    }
    if matches!(block.id.as_str(), "redstone_torch" | "redstone_wall_torch") {
        return true;
    }
    if matches!(block.id.as_str(), "daylight_detector" | "redstone_wire") {
        return block
            .attributes
            .get("power")
            .and_then(|value| value.parse::<i32>().ok())
            .unwrap_or(0)
            > 0;
    }
    if matches!(block.id.as_str(), "lever")
        || block.id.contains("button")
        || block.id.contains("pressure_plate")
    {
        return bool_attr(block, "powered");
    }
    false
}

fn scaffolding_distance_attr(block: &Block) -> u8 {
    block
        .attributes
        .get("distance")
        .and_then(|value| value.parse::<u8>().ok())
        .unwrap_or(7)
}

fn scaffolding_distance(region: &Region, pos: [i32; 3]) -> u8 {
    let below_pos = offset_pos(pos, [0, -1, 0]);
    if let Some(below_block) = block_at(region, below_pos) {
        if below_block.namespace == "minecraft" && below_block.id == "scaffolding" {
            return scaffolding_distance_attr(below_block);
        }
        if is_face_sturdy(below_block, FaceDirection::Up) {
            return 0;
        }
    }

    let mut distance = 7_u8;
    for offset in [[-1, 0, 0], [1, 0, 0], [0, 0, -1], [0, 0, 1]] {
        let neighbor_pos = offset_pos(pos, offset);
        let Some(neighbor_block) = block_at(region, neighbor_pos) else {
            continue;
        };
        if neighbor_block.namespace != "minecraft" || neighbor_block.id != "scaffolding" {
            continue;
        }
        distance = distance.min(scaffolding_distance_attr(neighbor_block).saturating_add(1));
        if distance == 1 {
            break;
        }
    }
    distance
}

fn scaffolding_is_bottom(region: &Region, pos: [i32; 3], distance: u8) -> bool {
    if distance == 0 {
        return false;
    }
    block_at(region, offset_pos(pos, [0, -1, 0]))
        .map(|block| !(block.namespace == "minecraft" && block.id == "scaffolding"))
        .unwrap_or(true)
}

fn is_waterloggable_block(block: &Block) -> bool {
    if block.namespace != "minecraft"
        || block.id.ends_with("fence_gate")
        || block.id.ends_with("door")
        || block.id.ends_with("pressure_plate")
        || block.id == "end_rod"
    {
        return false;
    }

    block.id.ends_with("slab")
        || block.id.ends_with("stairs")
        || block.id.ends_with("trapdoor")
        || block.id.ends_with("fence")
        || block.id.ends_with("pane")
        || block.id.ends_with("bars")
        || block.id.ends_with("wall")
        || block.id == "ladder"
        || block.id.ends_with("_chain")
        || block.id == "lightning_rod"
        || block.id == "pointed_dripstone"
        || matches!(block.id.as_str(), "big_dripleaf" | "big_dripleaf_stem")
        || block.id == "scaffolding"
        || block.id == "decorated_pot"
        || matches!(block.id.as_str(), "chest" | "trapped_chest")
        || block.id.ends_with("sign")
}

fn slab_is_double(block: &Block) -> bool {
    block.id.ends_with("slab") && block.attributes.get("type").map(String::as_str) == Some("double")
}

fn is_waterlogged(block: &Block) -> bool {
    block.attributes.get("waterlogged").map(String::as_str) == Some("true")
}

fn with_waterlogged(block: &Block, waterlogged: bool) -> Block {
    with_attr(
        block,
        "waterlogged",
        if waterlogged { "true" } else { "false" },
    )
}
fn try_place_waterlogged_block(
    region: &mut Region,
    pos: [i32; 3],
    block: &Block,
    fluid: DynamicWaterState,
) -> bool {
    if !can_place_water_in_block(block, fluid) {
        return false;
    }
    let updated = with_waterlogged(block, true);
    let _ = region.set_block(pos, &updated);
    true
}

fn can_place_water_in_block(block: &Block, fluid: DynamicWaterState) -> bool {
    fluid.is_source()
        && is_waterloggable_block(block)
        && !is_waterlogged(block)
        && !slab_is_double(block)
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

fn can_hold_specific_fluid(block: &Block, fluid: DynamicWaterState) -> bool {
    !is_waterloggable_block(block) || can_place_water_in_block(block, fluid)
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
    alive: bool,
    removed_by: String,
    on_fire: bool,
    remaining_fire_ticks: i32,
    health: Option<i32>,
    item_health: Option<i32>,
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
    pushout_speed: f64,
    in_water: bool,
    underwater_movement: bool,
    in_lava: bool,
    underlava_movement: bool,
    fluid_height: f64,
    water_height: f64,
    lava_height: f64,
    active_fluid: String,
    active_fluid_height: f64,
    fall_distance: f64,
    current_samples: usize,
    applied_current_x: f64,
    applied_current_y: f64,
    applied_current_z: f64,
    water_current_samples: usize,
    lava_current_samples: usize,
    applied_water_current_x: f64,
    applied_water_current_y: f64,
    applied_water_current_z: f64,
    applied_lava_current_x: f64,
    applied_lava_current_y: f64,
    applied_lava_current_z: f64,
    block_x: i32,
    block_y: i32,
    block_z: i32,
    support_block: String,
    center_block: String,
}

#[derive(Clone, Debug)]
pub(crate) struct VerifyTracePoint {
    pub x: f64,
    pub y: f64,
    pub vx: f64,
    pub vy: f64,
    pub on_ground: bool,
    pub support_block: String,
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
    in_lava_ratio: f64,
    underlava_ratio: f64,
    alive_ratio: f64,
    on_fire_ratio: f64,
    removal_tick: Option<usize>,
    removal_reason: String,
    first_on_fire_tick: Option<usize>,
    first_in_water_tick: Option<usize>,
    first_underwater_tick: Option<usize>,
    first_in_lava_tick: Option<usize>,
    first_underlava_tick: Option<usize>,
    unsupported_collision_block_count: usize,
    unsupported_collision_blocks: Vec<String>,
    no_physics_ratio: f64,
    pushout_tick_count: usize,
    collision_model: &'static str,
    fluid_model: &'static str,
    fluid_schedule_mode: &'static str,
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
    lava_flow_scale: f64,
    horizontal_water_damping: f64,
    horizontal_lava_damping: f64,
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
    no_physics_pushout_speed_min: f64,
    no_physics_pushout_speed_max: f64,
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

#[derive(Clone, Debug)]
pub(crate) struct VerifyRunOutput {
    pub world_name: String,
    pub shape: [i32; 3],
    pub tick_csv_path: PathBuf,
    pub summary_csv_path: PathBuf,
    pub summary_json_path: PathBuf,
    inspect_tick: Option<VerifyTickRow>,
    pub approximate_collision_block_count: usize,
}

fn execute_verify_command(command: &VerifyCommand) -> Result<VerifyRunOutput, String> {
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

    Ok(VerifyRunOutput {
        world_name: world.name,
        shape: world.region.shape(),
        tick_csv_path,
        summary_csv_path,
        summary_json_path,
        inspect_tick,
        approximate_collision_block_count: world.approximate_collision_blocks.len(),
    })
}

pub(crate) fn run_verify_command_quiet(command: &VerifyCommand) -> Result<VerifyRunOutput, String> {
    execute_verify_command(command)
}

pub(crate) fn run_verify_command(command: &VerifyCommand) -> Result<(), String> {
    let output = execute_verify_command(command)?;
    println!("World: {}", output.world_name);
    println!(
        "Shape: {} x {} x {}",
        output.shape[0], output.shape[1], output.shape[2]
    );
    println!("Tick CSV: {}", output.tick_csv_path.display());
    println!("Summary CSV: {}", output.summary_csv_path.display());
    println!("Summary JSON: {}", output.summary_json_path.display());
    if let Some(row) = output.inspect_tick {
        println!();
        println!(
            "Tick {} => pos=({:.6}, {:.6}, {:.6}) vel=({:.6}, {:.6}, {:.6}) fluid={} water={} lava={} fluidHeight={:.6} onFire={} fireTicks={} itemHealth={}",
            row.tick,
            row.x,
            row.y,
            row.z,
            row.vx,
            row.vy,
            row.vz,
            row.active_fluid,
            row.in_water,
            row.in_lava,
            row.fluid_height,
            row.on_fire,
            row.remaining_fire_ticks,
            row.item_health
                .map(|value| value.to_string())
                .unwrap_or_else(|| "n/a".to_string()),
        );
    }
    if output.approximate_collision_block_count > 0 {
        println!();
        println!(
            "Warning: {} block types use approximate non-solid collision handling.",
            output.approximate_collision_block_count
        );
    }
    Ok(())
}

pub(crate) fn simulate_verify_trace(
    world: &LoadedSchematic,
    command: &VerifyCommand,
) -> Vec<VerifyTracePoint> {
    simulate(world, command)
        .into_iter()
        .map(|row| VerifyTracePoint {
            x: row.x,
            y: row.y,
            vx: row.vx,
            vy: row.vy,
            on_ground: row.on_ground,
            support_block: row.support_block,
        })
        .collect()
}

fn simulate(world: &LoadedSchematic, command: &VerifyCommand) -> Vec<VerifyTickRow> {
    let mut region = world.region.clone();
    normalize_loaded_snapshot(&mut region);
    normalize_loaded_motion_snapshot(&mut region);
    let mut block_ticks = DynamicBlockTicks::bootstrap(&region);
    let mut fluid_ticks = if command.bootstrap_fluids {
        DynamicFluidTicks::bootstrap(&region)
    } else {
        DynamicFluidTicks::default()
    };
    let hopper_positions = collect_hopper_positions(&region);
    let mut entity_rng = command.entity_rng_seed.map(LegacyRandom::new).or_else(|| {
        command
            .entity_uuid
            .as_deref()
            .and_then(legacy_random_from_entity_uuid)
    });
    let mut rows = Vec::with_capacity(command.ticks + 1);
    let start_pos = Vec3d::new(command.start_x, command.start_y, command.start_z);
    let mut pos = start_pos;
    let mut vel = Vec3d::new(command.start_vx, command.start_vy, command.start_vz);
    let mut on_ground = command.start_on_ground;
    let mut alive = true;
    let mut removed_by: Option<&'static str> = None;
    let mut stuck_speed_multiplier = Vec3d::ZERO;
    let mut fall_distance = 0.0;
    let mut remaining_fire_ticks = command.start_fire_ticks;
    let mut item_health = command
        .item_health
        .or_else(|| default_item_health(command.entity_kind));
    let forward_axis = forward_basis(vel);
    let lateral_axis = Vec3d::new(-forward_axis.z, 0.0, forward_axis.x);
    let initial_water = track_water(&region, pos, command.width, command.height, false);
    let initial_lava = track_lava(&region, pos, command.width, command.height, false);
    let initial_no_physics = is_no_physics(
        &region,
        pos,
        command.width,
        command.height,
        fall_distance,
        command.entity_kind,
    );

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
        0.0,
        initial_water,
        initial_lava,
        Vec3d::ZERO,
        Vec3d::ZERO,
        command.entity_kind,
        forward_axis,
        lateral_axis,
        start_pos,
        command.height,
        &region,
        alive,
        removed_by.unwrap_or(""),
        fall_distance,
        remaining_fire_ticks,
        item_health,
    ));

    for tick in 1..=command.ticks {
        block_ticks.run_due(&mut region, tick);
        fluid_ticks.run_due(&mut region, &mut block_ticks, tick);

        if !alive {
            let _ = tick_hoppers(&mut region, &hopper_positions, None);
            rows.push(build_tick_row(
                tick,
                pos,
                vel,
                on_ground,
                false,
                false,
                false,
                false,
                false,
                false,
                0.0,
                FluidTracker::default(),
                FluidTracker::default(),
                Vec3d::ZERO,
                Vec3d::ZERO,
                command.entity_kind,
                forward_axis,
                lateral_axis,
                start_pos,
                command.height,
                &region,
                false,
                removed_by.unwrap_or(""),
                fall_distance,
                remaining_fire_ticks,
                item_health,
            ));
            continue;
        }

        let pre_move_water = track_water(&region, pos, command.width, command.height, false);
        let pre_move_lava = track_lava(&region, pos, command.width, command.height, false);
        tick_fire_state(
            pre_move_lava.is_in_fluid(),
            command.fire_immune,
            &mut remaining_fire_ticks,
            &mut item_health,
            &mut alive,
            &mut removed_by,
        );
        if pre_move_lava.is_in_fluid() {
            fall_distance *= 0.5;
        }
        if !alive {
            rows.push(build_tick_row(
                tick,
                pos,
                vel,
                on_ground,
                false,
                false,
                false,
                false,
                false,
                false,
                0.0,
                pre_move_water,
                pre_move_lava,
                Vec3d::ZERO,
                Vec3d::ZERO,
                command.entity_kind,
                forward_axis,
                lateral_axis,
                start_pos,
                command.height,
                &region,
                false,
                removed_by.unwrap_or(""),
                fall_distance,
                remaining_fire_ticks,
                item_health,
            ));
            continue;
        }
        if on_ground
            && !entity_has_ground_support(
                &region,
                pos,
                command.width,
                command.height,
                fall_distance,
                command.entity_kind,
            )
        {
            on_ground = false;
        }
        let mut applied_water_current = if pre_move_water.is_in_fluid() {
            apply_water_current(&mut vel, pre_move_water)
        } else {
            Vec3d::ZERO
        };
        let mut applied_lava_current = if pre_move_lava.is_in_fluid() {
            apply_lava_current(&mut vel, pre_move_lava)
        } else {
            Vec3d::ZERO
        };
        apply_living_delta_deadzone(command.entity_kind, &mut vel);

        let skips_travel = skips_active_travel(command.entity_kind, command.no_ai);
        let active_living =
            matches!(command.entity_kind, VerifyEntityKind::Living) && !command.no_ai;
        let living_fluid_falling = active_living
            && (pre_move_water.is_in_fluid() || pre_move_lava.is_in_fluid())
            && vel.y <= 0.0;
        if active_living && !pre_move_water.is_in_fluid() && !pre_move_lava.is_in_fluid() {
            apply_living_climbable_motion(&region, pos, &mut vel, &mut fall_distance);
        }
        if !skips_travel {
            if active_living {
            } else if pre_move_water.is_in_fluid() && pre_move_water.applies_movement_damping() {
                vel.x *= HORIZONTAL_WATER_DAMPING;
                vel.z *= HORIZONTAL_WATER_DAMPING;
                if vel.y < BUOYANCY_CAP {
                    vel.y += BUOYANCY;
                }
            } else if pre_move_lava.is_in_fluid() && pre_move_lava.applies_movement_damping() {
                vel.x *= HORIZONTAL_LAVA_DAMPING;
                vel.z *= HORIZONTAL_LAVA_DAMPING;
                if vel.y < BUOYANCY_CAP {
                    vel.y += BUOYANCY;
                }
            } else {
                vel.y -= base_gravity(command.entity_kind, command.no_gravity);
            }
        }

        let no_physics = is_no_physics(
            &region,
            pos,
            command.width,
            command.height,
            fall_distance,
            command.entity_kind,
        );
        let (pushout_applied, pushout_speed) = if no_physics {
            let (adjusted, speed) =
                move_towards_closest_space(&region, pos, vel, command.height, entity_rng.as_mut());
            vel = adjusted;
            (true, speed)
        } else {
            (false, 0.0)
        };

        let phase_mod4 =
            (command.initial_tick_count + tick + command.entity_id_mod4) % MOVEMENT_SAMPLE_MODULO;
        let should_move = if skips_travel {
            false
        } else if uses_item_movement_sampling(command.entity_kind) {
            !on_ground
                || vel.horizontal_length_sqr() > HORIZONTAL_REST_THRESHOLD2
                || phase_mod4 == 0
        } else {
            !on_ground || vel.length_sqr() > 1.0e-18
        };
        let mut moved = false;
        let mut collided_x = false;
        let mut collided_y = false;
        let mut collided_z = false;
        let mut actual_on_ground = on_ground;
        let from_pos = pos;

        if should_move {
            let landed;
            if no_physics {
                pos = pos.add(vel);
                moved = vel.length_sqr() > 1.0e-18;
                actual_on_ground = on_ground;
                landed = false;
            } else {
                let move_delta = if stuck_speed_multiplier.length_sqr() > 1.0e-7 {
                    let scaled = Vec3d::new(
                        vel.x * stuck_speed_multiplier.x,
                        vel.y * stuck_speed_multiplier.y,
                        vel.z * stuck_speed_multiplier.z,
                    );
                    stuck_speed_multiplier = Vec3d::ZERO;
                    vel = Vec3d::ZERO;
                    scaled
                } else {
                    vel
                };
                let move_result = move_entity_with_fall_distance(
                    &region,
                    pos,
                    move_delta,
                    command.width,
                    command.height,
                    fall_distance,
                    command.entity_kind,
                );
                if fall_distance != 0.0
                    && movement_resets_fall_distance(&region, pos, move_result.delta)
                {
                    fall_distance = 0.0;
                }
                pos = pos.add(move_result.delta);
                moved = move_result.delta.length_sqr() > 1.0e-18;
                collided_x = move_result.collided_x;
                collided_y = move_result.collided_y;
                collided_z = move_result.collided_z;
                if collided_x {
                    vel.x = 0.0;
                }
                if collided_z {
                    vel.z = 0.0;
                }
                if collided_y && vel.y > 0.0 {
                    vel.y = 0.0;
                }
                landed = move_result.collided_y && vel.y < 0.0;
                actual_on_ground = landed
                    || (on_ground
                        && entity_has_ground_support(
                            &region,
                            pos,
                            command.width,
                            command.height,
                            fall_distance,
                            command.entity_kind,
                        ));
            }
            if landed && !active_living {
                vel.y = vertical_velocity_after_landing(
                    ground_profile_at_for(&region, pos, command.entity_kind).slime_surface,
                    vel.y,
                    pos.y - from_pos.y,
                    base_gravity(command.entity_kind, command.no_gravity),
                    command.entity_kind,
                );
            }
            if !no_physics {
                let post_move_on_climbable = active_living && living_on_climbable(&region, pos);
                if active_living && (collided_x || collided_z) && post_move_on_climbable {
                    vel.y = LIVING_CLIMBABLE_ASCENT;
                }
                if !pre_move_water.is_in_fluid() && pos.y < from_pos.y {
                    fall_distance += from_pos.y - pos.y;
                }
                if actual_on_ground {
                    apply_fall_on_effects(
                        &region,
                        pos,
                        command.entity_kind,
                        command.fire_immune,
                        fall_distance,
                        &mut item_health,
                        &mut alive,
                        &mut removed_by,
                    );
                    fall_distance = 0.0;
                }
            }
            if !no_physics {
                let speed_factor = block_speed_factor_for(&region, pos, command.entity_kind);
                vel.x *= speed_factor;
                vel.z *= speed_factor;
                if active_living {
                    if pre_move_water.is_in_fluid() {
                        apply_living_water_physics(
                            &mut vel,
                            base_gravity(command.entity_kind, command.no_gravity),
                            living_fluid_falling,
                        );
                        maybe_jump_living_out_of_fluid(
                            &region,
                            pos,
                            from_pos.y,
                            command.width,
                            command.height,
                            fall_distance,
                            &mut vel,
                            collided_x || collided_z,
                        );
                    } else if pre_move_lava.is_in_fluid() {
                        apply_living_lava_physics(
                            &mut vel,
                            base_gravity(command.entity_kind, command.no_gravity),
                            living_fluid_falling,
                            pre_move_lava.height,
                            command.height,
                        );
                        maybe_jump_living_out_of_fluid(
                            &region,
                            pos,
                            from_pos.y,
                            command.width,
                            command.height,
                            fall_distance,
                            &mut vel,
                            collided_x || collided_z,
                        );
                    } else {
                        apply_living_air_physics(
                            &mut vel,
                            actual_on_ground,
                            ground_profile_at_for(&region, pos, command.entity_kind),
                            base_gravity(command.entity_kind, command.no_gravity),
                        );
                    }
                }
            }
            if alive && actual_on_ground {
                apply_step_on_effects(
                    &region,
                    pos,
                    command.entity_kind,
                    moved,
                    command.fire_immune,
                    &mut item_health,
                    &mut alive,
                    &mut removed_by,
                );
            }
            if alive && !no_physics {
                if let Some(reason) = apply_inside_block_effects(
                    &region,
                    from_pos,
                    pos,
                    command.width,
                    command.height,
                    command.entity_kind,
                    command.fire_immune,
                    actual_on_ground,
                    &mut vel,
                    &mut stuck_speed_multiplier,
                    &mut fall_distance,
                    &mut remaining_fire_ticks,
                    &mut item_health,
                ) {
                    alive = false;
                    removed_by = Some(reason);
                }
            }
            if !active_living {
                let drag = horizontal_drag(
                    ground_profile_at_for(&region, pos, command.entity_kind),
                    actual_on_ground,
                    vel.y,
                );
                vel.x *= drag;
                vel.z *= drag;
                vel.y *= VERTICAL_MOVEMENT_DAMPING;
                if actual_on_ground && vel.y < 0.0 {
                    vel.y *= -0.5;
                }
            }
        } else {
            if !no_physics {
                if let Some(reason) = apply_inside_block_effects(
                    &region,
                    pos,
                    pos,
                    command.width,
                    command.height,
                    command.entity_kind,
                    command.fire_immune,
                    on_ground,
                    &mut vel,
                    &mut stuck_speed_multiplier,
                    &mut fall_distance,
                    &mut remaining_fire_ticks,
                    &mut item_health,
                ) {
                    alive = false;
                    removed_by = Some(reason);
                }
            }
        }

        if alive {
            maybe_trigger_big_dripleaf(
                &mut region,
                &mut block_ticks,
                pos,
                command.width,
                actual_on_ground,
                tick,
            );
        }

        let post_move_water = track_water(&region, pos, command.width, command.height, false);
        let post_move_lava = track_lava(&region, pos, command.width, command.height, false);
        if has_post_move_fluid_current(command.entity_kind) {
            if post_move_water.is_in_fluid() {
                applied_water_current =
                    applied_water_current.add(apply_water_current(&mut vel, post_move_water));
            }
            if post_move_lava.is_in_fluid() {
                applied_lava_current =
                    applied_lava_current.add(apply_lava_current(&mut vel, post_move_lava));
            }
        }

        let hopper_box = if hopper_collects_entity(command.entity_kind) && alive {
            Some(entity_aabb(pos, command.width, command.height))
        } else {
            None
        };
        if let Some(reason) = tick_hoppers(&mut region, &hopper_positions, hopper_box) {
            alive = false;
            removed_by = Some(reason);
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
            pushout_speed,
            post_move_water,
            post_move_lava,
            applied_water_current,
            applied_lava_current,
            command.entity_kind,
            forward_axis,
            lateral_axis,
            start_pos,
            command.height,
            &region,
            alive,
            removed_by.unwrap_or(""),
            fall_distance,
            remaining_fire_ticks,
            item_health,
        ));
        on_ground = actual_on_ground;
    }

    rows
}

// Match observed vanilla save/reload semantics for schematic-like static worlds.
// Some unsupported blocks are pruned before the first active tick after load,
// while others persist and only react to later neighbor updates.
fn normalize_loaded_snapshot(region: &mut Region) {
    loop {
        let shape = region.shape();
        let mut removals = Vec::new();
        for x in 0..shape[0] {
            for y in 0..shape[1] {
                for z in 0..shape[2] {
                    let pos = [x, y, z];
                    let Some(block) = block_at(region, pos) else {
                        continue;
                    };
                    if snapshot_block_is_pruned_on_reload(region, pos, block) {
                        removals.push(pos);
                    }
                }
            }
        }
        if removals.is_empty() {
            return;
        }
        for pos in removals {
            set_air_block(region, pos);
        }
    }
}

fn normalize_loaded_motion_snapshot(region: &mut Region) {
    let shape = region.shape();
    for x in 0..shape[0] {
        for y in 0..shape[1] {
            for z in 0..shape[2] {
                let pos = [x, y, z];
                let Some(block) = block_at(region, pos).cloned() else {
                    continue;
                };
                if snapshot_bubble_column_is_inactive_on_reload(region, pos, &block) {
                    set_water_block(
                        region,
                        pos,
                        DynamicWaterState {
                            amount: 8,
                            falling: false,
                        },
                    );
                }
            }
        }
    }
}

fn snapshot_block_is_pruned_on_reload(region: &Region, pos: [i32; 3], block: &Block) -> bool {
    if block.namespace != "minecraft" {
        return false;
    }
    match block.id.as_str() {
        "bamboo" => !bamboo_can_survive(region, pos),
        "cactus" => !cactus_can_survive(region, pos),
        "scaffolding" => scaffolding_distance(region, pos) == 7,
        "pointed_dripstone" => !pointed_dripstone_can_survive(region, pos, block),
        "big_dripleaf_stem" => !big_dripleaf_stem_can_survive(region, pos),
        "big_dripleaf" => snapshot_big_dripleaf_leaf_is_pruned_on_reload(region, pos),
        _ => false,
    }
}

fn snapshot_bubble_column_is_inactive_on_reload(
    region: &Region,
    pos: [i32; 3],
    block: &Block,
) -> bool {
    if block.namespace != "minecraft" || block.id != "bubble_column" {
        return false;
    }
    !matches!(
        bubble_column_state_from_below(region, pos, block),
        Some(expected) if expected.id == "bubble_column" && expected.full_id() == block.full_id()
    )
}

fn snapshot_big_dripleaf_leaf_is_pruned_on_reload(region: &Region, pos: [i32; 3]) -> bool {
    let below_pos = offset_pos(pos, [0, -1, 0]);
    block_at(region, below_pos)
        .filter(|below| below.namespace == "minecraft" && below.id == "big_dripleaf_stem")
        .map(|_| !big_dripleaf_stem_can_survive(region, below_pos))
        .unwrap_or(false)
}

fn collect_hopper_positions(region: &Region) -> Vec<[i32; 3]> {
    let shape = region.shape();
    let mut result = Vec::new();
    for x in 0..shape[0] {
        for y in 0..shape[1] {
            for z in 0..shape[2] {
                let pos = [x, y, z];
                let Some(block) = block_at(region, pos) else {
                    continue;
                };
                if block.namespace == "minecraft" && block.id == "hopper" {
                    result.push(pos);
                }
            }
        }
    }
    result
}

fn tick_hoppers(
    region: &mut Region,
    hopper_positions: &[[i32; 3]],
    entity_box: Option<Aabb>,
) -> Option<&'static str> {
    let mut item_box = entity_box;
    let mut removal_reason = None;
    for &hopper_pos in hopper_positions {
        if let Some(reason) = tick_single_hopper(region, hopper_pos, item_box) {
            removal_reason.get_or_insert(reason);
            item_box = None;
        }
    }
    removal_reason
}

fn tick_single_hopper(
    region: &mut Region,
    pos: [i32; 3],
    entity_box: Option<Aabb>,
) -> Option<&'static str> {
    let Some(block) = block_at(region, pos) else {
        return None;
    };
    if block.namespace != "minecraft" || block.id != "hopper" || !block_enabled(block) {
        return None;
    }

    let cooldown = hopper_cooldown(region, pos).unwrap_or(-1) - 1;
    set_hopper_cooldown(region, pos, cooldown.max(0));
    if cooldown > 0 {
        return None;
    }
    if hopper_inventory_full(region, pos) {
        return None;
    }

    let Some(item_box) = entity_box else {
        return None;
    };
    if hopper_air_suction_blocked(region, pos) || has_source_container_above(region, pos) {
        return None;
    }
    if !aabbs_intersect(item_box, hopper_suck_aabb_world(pos)) {
        return None;
    }

    set_hopper_cooldown(region, pos, 8);
    Some("hopperTickSuck")
}

fn apply_inside_block_effects(
    region: &Region,
    from_pos: Vec3d,
    to_pos: Vec3d,
    width: f64,
    height: f64,
    entity_kind: VerifyEntityKind,
    fire_immune: bool,
    on_ground: bool,
    velocity: &mut Vec3d,
    stuck_speed_multiplier: &mut Vec3d,
    fall_distance: &mut f64,
    remaining_fire_ticks: &mut i32,
    item_health: &mut Option<i32>,
) -> Option<&'static str> {
    let final_box = entity_aabb(to_pos, width, height);
    let swept_box = swept_entity_aabb(from_pos, to_pos, width, height);
    let from_box = entity_aabb(from_pos, width, height);
    let movement = Vec3d::new(
        to_pos.x - from_pos.x,
        to_pos.y - from_pos.y,
        to_pos.z - from_pos.z,
    );
    let x0 = swept_box.min_x.floor() as i32;
    let x1 = swept_box.max_x.ceil() as i32 - 1;
    let y0 = swept_box.min_y.floor() as i32;
    let y1 = swept_box.max_y.ceil() as i32 - 1;
    let z0 = swept_box.min_z.floor() as i32;
    let z1 = swept_box.max_z.ceil() as i32 - 1;
    let mut applied_honey_slide = false;
    let mut applied_bubble_column = false;
    let mut touched_water = false;
    let mut touched_lava = false;
    let mut touched_cactus = false;
    let mut campfire_damage = None;
    let mut fire_damage = None;
    let in_block_powder_snow = block_at(
        region,
        [
            to_pos.x.floor() as i32,
            to_pos.y.floor() as i32,
            to_pos.z.floor() as i32,
        ],
    )
    .map(|block| block.namespace == "minecraft" && block.id == "powder_snow")
    .unwrap_or(false);

    for x in x0..=x1 {
        for y in y0..=y1 {
            for z in z0..=z1 {
                let block_pos = [x, y, z];
                if !applied_honey_slide
                    && final_box_intersects_block(final_box, block_pos)
                    && maybe_apply_honey_slide(
                        region,
                        block_pos,
                        to_pos,
                        width,
                        on_ground,
                        velocity,
                        fall_distance,
                    )
                {
                    applied_honey_slide = true;
                }
                if !applied_bubble_column
                    && final_box_intersects_block(final_box, block_pos)
                    && maybe_apply_bubble_column(region, block_pos, velocity, fall_distance)
                {
                    applied_bubble_column = true;
                }
                if final_box_intersects_block(final_box, block_pos) {
                    maybe_apply_stuck_block(
                        region,
                        block_pos,
                        stuck_speed_multiplier,
                        fall_distance,
                        entity_kind,
                        in_block_powder_snow,
                    );
                }
                if !touched_water
                    && movement_box_intersects_fluid(
                        region,
                        from_box,
                        movement,
                        block_pos,
                        FluidKind::Water,
                    )
                {
                    touched_water = true;
                }
                if !touched_lava
                    && movement_box_intersects_fluid(
                        region,
                        from_box,
                        movement,
                        block_pos,
                        FluidKind::Lava,
                    )
                {
                    touched_lava = true;
                }
                if !touched_lava
                    && movement_box_intersects_lava_cauldron(region, from_box, movement, block_pos)
                {
                    touched_lava = true;
                }
                if let Some(damage) =
                    movement_box_fire_damage(region, from_box, movement, block_pos)
                {
                    fire_damage =
                        Some(fire_damage.map_or(damage, |current: i32| current.max(damage)));
                }
                if !touched_cactus
                    && movement_box_intersects_cactus(region, from_box, movement, block_pos)
                {
                    touched_cactus = true;
                }
                if let Some(damage) =
                    movement_box_campfire_damage(region, from_box, movement, block_pos, entity_kind)
                {
                    campfire_damage =
                        Some(campfire_damage.map_or(damage, |current: i32| current.max(damage)));
                }
                if !hopper_collects_entity(entity_kind) {
                    continue;
                }
                if !hopper_can_collect_now(region, block_pos) {
                    continue;
                }
                if !final_box_intersects_block(final_box, block_pos) {
                    continue;
                }
                if aabbs_intersect(final_box, hopper_suck_aabb_world(block_pos)) {
                    return Some("hopperEntityInside");
                }
            }
        }
    }

    if let Some(damage) = fire_damage {
        fire_ignite(remaining_fire_ticks, fire_immune);
        let mut alive = true;
        let mut removed_by = None;
        on_fire_hurt_with_damage(
            fire_immune,
            item_health,
            damage,
            &mut alive,
            &mut removed_by,
        );
        if let Some(reason) = removed_by {
            return Some(reason);
        }
    }
    if touched_lava {
        lava_ignite(remaining_fire_ticks, fire_immune);
        let mut alive = true;
        let mut removed_by = None;
        lava_hurt(fire_immune, item_health, &mut alive, &mut removed_by);
        if let Some(reason) = removed_by {
            return Some(reason);
        }
    }
    if touched_water {
        clear_fire(remaining_fire_ticks);
    }
    if touched_cactus {
        let mut alive = true;
        let mut removed_by = None;
        damage_tracked_entity(item_health, 1, "cactusDamage", &mut alive, &mut removed_by);
        if let Some(reason) = removed_by {
            return Some(reason);
        }
    }
    if let Some(damage) = campfire_damage {
        let mut alive = true;
        let mut removed_by = None;
        damage_tracked_entity(
            item_health,
            damage,
            "campfireDamage",
            &mut alive,
            &mut removed_by,
        );
        if let Some(reason) = removed_by {
            return Some(reason);
        }
    }

    None
}

fn on_pos_legacy_block_pos(pos: Vec3d) -> [i32; 3] {
    [
        pos.x.floor() as i32,
        (pos.y - ON_POS_LEGACY_OFFSET).floor() as i32,
        pos.z.floor() as i32,
    ]
}

fn apply_step_on_effects(
    region: &Region,
    pos: Vec3d,
    entity_kind: VerifyEntityKind,
    moved: bool,
    fire_immune: bool,
    health: &mut Option<i32>,
    alive: &mut bool,
    removed_by: &mut Option<&'static str>,
) {
    if !moved || !matches!(entity_kind, VerifyEntityKind::Living) {
        return;
    }
    let Some(block) = block_at(region, on_pos_legacy_block_pos(pos)) else {
        return;
    };
    if fire_immune {
        return;
    }
    if block.namespace == "minecraft" && block.id == "magma_block" {
        damage_tracked_entity(health, 1, "hotFloorDamage", alive, removed_by);
    }
}

fn pointed_dripstone_fall_damage(fall_distance: f64) -> i32 {
    (((fall_distance + 2.5) + 1.0e-6 - 3.0).max(0.0) * 2.0).floor() as i32
}

fn apply_fall_on_effects(
    region: &Region,
    pos: Vec3d,
    entity_kind: VerifyEntityKind,
    fire_immune: bool,
    fall_distance: f64,
    health: &mut Option<i32>,
    alive: &mut bool,
    removed_by: &mut Option<&'static str>,
) {
    if fall_distance <= 0.0 || !matches!(entity_kind, VerifyEntityKind::Living) {
        return;
    }
    let Some(block) = block_at(region, on_pos_legacy_block_pos(pos)) else {
        return;
    };
    if block.namespace != "minecraft" || block.id != "pointed_dripstone" {
        return;
    }
    if block
        .attributes
        .get("vertical_direction")
        .map(String::as_str)
        != Some("up")
        || block.attributes.get("thickness").map(String::as_str) != Some("tip")
    {
        return;
    }
    let damage = pointed_dripstone_fall_damage(fall_distance);
    if damage == 0 {
        return;
    }
    if fire_immune {
        return;
    }
    damage_tracked_entity(health, damage, "stalagmiteDamage", alive, removed_by);
}

fn final_box_intersects_block(final_box: Aabb, block_pos: [i32; 3]) -> bool {
    aabbs_intersect(final_box, full_block_aabb(block_pos))
}

fn movement_box_intersects_fluid(
    region: &Region,
    from_box: Aabb,
    movement: Vec3d,
    block_pos: [i32; 3],
    fluid_kind: FluidKind,
) -> bool {
    let Some(fluid_cell) = fluid_at(region, block_pos, fluid_kind) else {
        return false;
    };
    moving_aabb_intersects_aabb(from_box, movement, fluid_aabb(block_pos, fluid_cell.height))
}

fn movement_box_fire_damage(
    region: &Region,
    from_box: Aabb,
    movement: Vec3d,
    block_pos: [i32; 3],
) -> Option<i32> {
    let block = block_at(region, block_pos)?;
    let damage = match (block.namespace.as_str(), block.id.as_str()) {
        ("minecraft", "fire") => 1,
        ("minecraft", "soul_fire") => 2,
        _ => return None,
    };
    moving_aabb_intersects_aabb(from_box, movement, fire_inside_effect_aabb(block_pos))
        .then_some(damage)
}

fn movement_box_intersects_lava_cauldron(
    region: &Region,
    from_box: Aabb,
    movement: Vec3d,
    block_pos: [i32; 3],
) -> bool {
    let Some(block) = block_at(region, block_pos) else {
        return false;
    };
    if block.namespace != "minecraft" || block.id != "lava_cauldron" {
        return false;
    }
    if moving_aabb_intersects_aabb(from_box, movement, lava_cauldron_content_aabb(block_pos)) {
        return true;
    }
    collision_boxes(block).iter().any(|shape| {
        moving_aabb_intersects_aabb(
            from_box,
            movement,
            world_collision_box(block, block_pos, shape),
        )
    })
}

fn movement_box_intersects_cactus(
    region: &Region,
    from_box: Aabb,
    movement: Vec3d,
    block_pos: [i32; 3],
) -> bool {
    let Some(block) = block_at(region, block_pos) else {
        return false;
    };
    if block.namespace != "minecraft" || block.id != "cactus" {
        return false;
    }
    collision_boxes(block).iter().any(|shape| {
        moving_aabb_intersects_aabb(
            from_box,
            movement,
            world_collision_box(block, block_pos, shape),
        )
    })
}

fn movement_box_campfire_damage(
    region: &Region,
    from_box: Aabb,
    movement: Vec3d,
    block_pos: [i32; 3],
    entity_kind: VerifyEntityKind,
) -> Option<i32> {
    if !matches!(entity_kind, VerifyEntityKind::Living) {
        return None;
    }
    let block = block_at(region, block_pos)?;
    let damage = match (block.namespace.as_str(), block.id.as_str()) {
        ("minecraft", "campfire") => 1,
        ("minecraft", "soul_campfire") => 2,
        _ => return None,
    };
    let lit = block
        .attributes
        .get("lit")
        .map(String::as_str)
        .unwrap_or("true")
        == "true";
    if !lit {
        return None;
    }
    collision_boxes(block)
        .iter()
        .any(|shape| {
            moving_aabb_intersects_aabb(
                from_box,
                movement,
                world_collision_box(block, block_pos, shape),
            )
        })
        .then_some(damage)
}

fn maybe_apply_honey_slide(
    region: &Region,
    block_pos: [i32; 3],
    pos: Vec3d,
    width: f64,
    on_ground: bool,
    velocity: &mut Vec3d,
    fall_distance: &mut f64,
) -> bool {
    let Some(block) = block_at(region, block_pos) else {
        return false;
    };
    if block.namespace != "minecraft" || block.id != "honey_block" {
        return false;
    }
    if !is_sliding_down_honey(block_pos, pos, *velocity, width, on_ground) {
        return false;
    }
    apply_honey_slide(velocity);
    *fall_distance = 0.0;
    true
}

fn maybe_apply_bubble_column(
    region: &Region,
    block_pos: [i32; 3],
    velocity: &mut Vec3d,
    fall_distance: &mut f64,
) -> bool {
    let Some(block) = block_at(region, block_pos) else {
        return false;
    };
    if block.namespace != "minecraft" || block.id != "bubble_column" {
        return false;
    }

    let drag_down = block
        .attributes
        .get("drag")
        .map(String::as_str)
        .unwrap_or("true")
        == "true";
    let above_pos = [block_pos[0], block_pos[1] + 1, block_pos[2]];
    let nothing_above = block_at(region, above_pos)
        .map(|above| collision_boxes(above).is_empty() && water_at_single_block(above).is_none())
        .unwrap_or(true);

    if nothing_above {
        apply_above_bubble_column(velocity, drag_down);
    } else {
        apply_inside_bubble_column(velocity, drag_down);
    }
    *fall_distance = 0.0;
    true
}

fn maybe_apply_stuck_block(
    region: &Region,
    block_pos: [i32; 3],
    stuck_speed_multiplier: &mut Vec3d,
    fall_distance: &mut f64,
    entity_kind: VerifyEntityKind,
    in_block_powder_snow: bool,
) {
    let Some(block) = block_at(region, block_pos) else {
        return;
    };
    if block.namespace != "minecraft" {
        return;
    }

    match block.id.as_str() {
        "cobweb" => {
            *stuck_speed_multiplier = Vec3d::new(0.25, 0.05_f32 as f64, 0.25);
            *fall_distance = 0.0;
        }
        "sweet_berry_bush" => {
            if !matches!(entity_kind, VerifyEntityKind::Living) {
                return;
            }
            let age = block
                .attributes
                .get("age")
                .and_then(|value| value.parse::<u8>().ok())
                .unwrap_or(0);
            if age == 0 {
                return;
            }
            *stuck_speed_multiplier = Vec3d::new(0.8_f32 as f64, 0.75, 0.8_f32 as f64);
            *fall_distance = 0.0;
        }
        "powder_snow" => {
            if !powder_snow_inside_effect_applies(entity_kind, in_block_powder_snow) {
                return;
            }
            *stuck_speed_multiplier = Vec3d::new(0.9_f32 as f64, 1.5, 0.9_f32 as f64);
            *fall_distance = 0.0;
        }
        _ => {}
    }
}

fn is_sliding_down_honey(
    block_pos: [i32; 3],
    pos: Vec3d,
    velocity: Vec3d,
    width: f64,
    on_ground: bool,
) -> bool {
    if on_ground {
        return false;
    }
    if pos.y > block_pos[1] as f64 + HONEY_SLIDE_TOP_Y - 1.0e-7 {
        return false;
    }
    if honey_old_delta_y(velocity.y) >= HONEY_SLIDE_MIN_OLD_DELTA_Y {
        return false;
    }
    let dx = (block_pos[0] as f64 + 0.5 - pos.x).abs();
    let dz = (block_pos[2] as f64 + 0.5 - pos.z).abs();
    let overlap_distance = 0.4375 + width * 0.5;
    dx + 1.0e-7 > overlap_distance || dz + 1.0e-7 > overlap_distance
}

fn apply_honey_slide(velocity: &mut Vec3d) {
    let old_delta_y = honey_old_delta_y(velocity.y);
    if old_delta_y < HONEY_SLIDE_STRONG_MIN_OLD_DELTA_Y {
        let horizontal_reduction_factor = HONEY_SLIDE_TARGET_OLD_DELTA_Y / old_delta_y;
        velocity.x *= horizontal_reduction_factor;
        velocity.z *= horizontal_reduction_factor;
    }
    velocity.y = honey_new_delta_y(HONEY_SLIDE_TARGET_OLD_DELTA_Y);
}

fn honey_old_delta_y(delta_y: f64) -> f64 {
    delta_y / VERTICAL_MOVEMENT_DAMPING + 0.08
}

fn honey_new_delta_y(delta_y: f64) -> f64 {
    (delta_y - 0.08) * VERTICAL_MOVEMENT_DAMPING
}

fn apply_above_bubble_column(velocity: &mut Vec3d, drag_down: bool) {
    velocity.y = if drag_down {
        (velocity.y - BUBBLE_COLUMN_DRAG_DOWN_ACCELERATION).max(-0.9)
    } else {
        (velocity.y + BUBBLE_COLUMN_SURFACE_ACCELERATION).min(1.8)
    };
}

fn apply_inside_bubble_column(velocity: &mut Vec3d, drag_down: bool) {
    velocity.y = if drag_down {
        (velocity.y - BUBBLE_COLUMN_DRAG_DOWN_ACCELERATION).max(-0.3)
    } else {
        (velocity.y + BUBBLE_COLUMN_INTERNAL_ACCELERATION).min(0.7)
    };
}

fn hopper_can_collect_now(region: &Region, pos: [i32; 3]) -> bool {
    let Some(block) = block_at(region, pos) else {
        return false;
    };
    if block.namespace != "minecraft" || block.id != "hopper" || !block_enabled(block) {
        return false;
    }
    if hopper_cooldown(region, pos).unwrap_or(-1) > 0 {
        return false;
    }
    !hopper_inventory_full(region, pos)
}

fn hopper_inventory_full(region: &Region, pos: [i32; 3]) -> bool {
    region
        .block_entity_at(pos)
        .map(hopper_inventory_full_approx)
        .unwrap_or(false)
}

fn hopper_inventory_full_approx(block_entity: &BlockEntity) -> bool {
    let Some(Value::List(items)) = block_entity.tags.get("Items") else {
        return false;
    };
    if items.len() < 5 {
        return false;
    }

    items.iter().all(|item| {
        let Value::Compound(compound) = item else {
            return true;
        };
        compound.get("Count").and_then(Value::as_i64).unwrap_or(64) >= 64
    })
}

fn block_entity_i64(block_entity: &BlockEntity, key: &str) -> Option<i64> {
    block_entity.tags.get(key).and_then(Value::as_i64)
}

fn hopper_cooldown(region: &Region, pos: [i32; 3]) -> Option<i32> {
    region
        .block_entity_at(pos)
        .and_then(|block_entity| block_entity_i64(block_entity, "TransferCooldown"))
        .and_then(|value| i32::try_from(value).ok())
}

fn set_hopper_cooldown(region: &mut Region, pos: [i32; 3], cooldown: i32) {
    let Some(block_entity) = region.block_entities.get_mut(&pos) else {
        return;
    };
    block_entity
        .tags
        .insert("TransferCooldown".to_string(), Value::Int(cooldown));
}

fn hopper_air_suction_blocked(region: &Region, pos: [i32; 3]) -> bool {
    let above_pos = [pos[0], pos[1] + 1, pos[2]];
    let Some(above_block) = block_at(region, above_pos) else {
        return false;
    };
    is_collision_shape_full_block(above_block) && !does_not_block_hoppers(above_block)
}

fn does_not_block_hoppers(block: &Block) -> bool {
    block.namespace == "minecraft" && matches!(block.id.as_str(), "bee_nest" | "beehive")
}

fn has_source_container_above(region: &Region, pos: [i32; 3]) -> bool {
    let above_pos = [pos[0], pos[1] + 1, pos[2]];
    let Some(above_block) = block_at(region, above_pos) else {
        return false;
    };
    is_source_container_block(above_block)
}

fn is_source_container_block(block: &Block) -> bool {
    if block.namespace != "minecraft" {
        return false;
    }
    if block.id == "composter"
        || block.id == "barrel"
        || block.id == "hopper"
        || block.id == "dispenser"
        || block.id == "dropper"
        || block.id == "brewing_stand"
        || block.id == "crafter"
        || block.id == "jukebox"
        || block.id == "decorated_pot"
        || block.id == "furnace"
        || block.id == "blast_furnace"
        || block.id == "smoker"
    {
        return true;
    }
    if block.id != "ender_chest" && block.id.contains("chest") {
        return true;
    }
    block.id.ends_with("shulker_box")
}

fn block_enabled(block: &Block) -> bool {
    block
        .attributes
        .get("enabled")
        .map(|value| value == "true")
        .unwrap_or(true)
}

fn swept_entity_aabb(from_pos: Vec3d, to_pos: Vec3d, width: f64, height: f64) -> Aabb {
    let from_box = entity_aabb(from_pos, width, height);
    let to_box = entity_aabb(to_pos, width, height);
    Aabb {
        min_x: from_box.min_x.min(to_box.min_x),
        min_y: from_box.min_y.min(to_box.min_y),
        min_z: from_box.min_z.min(to_box.min_z),
        max_x: from_box.max_x.max(to_box.max_x),
        max_y: from_box.max_y.max(to_box.max_y),
        max_z: from_box.max_z.max(to_box.max_z),
    }
}

fn full_block_aabb(block_pos: [i32; 3]) -> Aabb {
    Aabb {
        min_x: block_pos[0] as f64,
        min_y: block_pos[1] as f64,
        min_z: block_pos[2] as f64,
        max_x: block_pos[0] as f64 + 1.0,
        max_y: block_pos[1] as f64 + 1.0,
        max_z: block_pos[2] as f64 + 1.0,
    }
}

fn fluid_aabb(block_pos: [i32; 3], height: f64) -> Aabb {
    Aabb {
        min_x: block_pos[0] as f64,
        min_y: block_pos[1] as f64,
        min_z: block_pos[2] as f64,
        max_x: block_pos[0] as f64 + 1.0,
        max_y: block_pos[1] as f64 + height,
        max_z: block_pos[2] as f64 + 1.0,
    }
}

fn fire_inside_effect_aabb(block_pos: [i32; 3]) -> Aabb {
    Aabb {
        min_x: block_pos[0] as f64,
        min_y: block_pos[1] as f64,
        min_z: block_pos[2] as f64,
        max_x: block_pos[0] as f64 + 1.0,
        max_y: block_pos[1] as f64 + 1.0 / 16.0,
        max_z: block_pos[2] as f64 + 1.0,
    }
}

fn lava_cauldron_content_aabb(block_pos: [i32; 3]) -> Aabb {
    Aabb {
        min_x: block_pos[0] as f64 + 2.0 / 16.0,
        min_y: block_pos[1] as f64 + 4.0 / 16.0,
        min_z: block_pos[2] as f64 + 2.0 / 16.0,
        max_x: block_pos[0] as f64 + 14.0 / 16.0,
        max_y: block_pos[1] as f64 + 15.0 / 16.0,
        max_z: block_pos[2] as f64 + 14.0 / 16.0,
    }
}

fn hopper_suck_aabb_world(block_pos: [i32; 3]) -> Aabb {
    Aabb {
        min_x: block_pos[0] as f64,
        min_y: block_pos[1] as f64 + 11.0 / 16.0,
        min_z: block_pos[2] as f64,
        max_x: block_pos[0] as f64 + 1.0,
        max_y: block_pos[1] as f64 + 2.0,
        max_z: block_pos[2] as f64 + 1.0,
    }
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
    pushout_speed: f64,
    water: FluidTracker,
    lava: FluidTracker,
    applied_water_current: Vec3d,
    applied_lava_current: Vec3d,
    entity_kind: VerifyEntityKind,
    forward_axis: Vec3d,
    lateral_axis: Vec3d,
    start_pos: Vec3d,
    entity_height: f64,
    region: &Region,
    alive: bool,
    removed_by: &str,
    fall_distance: f64,
    remaining_fire_ticks: i32,
    item_health: Option<i32>,
) -> VerifyTickRow {
    let center_block = centered_block(region, pos, entity_height);
    let support = support_block_for(region, pos, entity_kind);
    let displacement = Vec3d::new(
        pos.x - start_pos.x,
        pos.y - start_pos.y,
        pos.z - start_pos.z,
    );
    let in_water = water.is_in_fluid();
    let in_lava = lava.is_in_fluid();
    let underwater_movement = water.applies_movement_damping();
    let underlava_movement = lava.applies_movement_damping();
    let active_fluid = if in_water {
        "water"
    } else if in_lava {
        "lava"
    } else {
        "none"
    };
    let active_fluid_height = if in_water {
        water.height
    } else if in_lava {
        lava.height
    } else {
        0.0
    };
    let applied_current = applied_water_current.add(applied_lava_current);
    VerifyTickRow {
        tick,
        alive,
        removed_by: removed_by.to_string(),
        on_fire: remaining_fire_ticks > 0,
        remaining_fire_ticks,
        health: item_health,
        item_health,
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
        pushout_speed,
        in_water,
        underwater_movement,
        in_lava,
        underlava_movement,
        fluid_height: active_fluid_height,
        water_height: water.height,
        lava_height: lava.height,
        active_fluid: active_fluid.to_string(),
        active_fluid_height,
        fall_distance,
        current_samples: water.current_count + lava.current_count,
        applied_current_x: applied_current.x,
        applied_current_y: applied_current.y,
        applied_current_z: applied_current.z,
        water_current_samples: water.current_count,
        lava_current_samples: lava.current_count,
        applied_water_current_x: applied_water_current.x,
        applied_water_current_y: applied_water_current.y,
        applied_water_current_z: applied_water_current.z,
        applied_lava_current_x: applied_lava_current.x,
        applied_lava_current_y: applied_lava_current.y,
        applied_lava_current_z: applied_lava_current.z,
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
    let mut in_lava_count = 0_usize;
    let mut underlava_count = 0_usize;
    let mut alive_count = 0_usize;
    let mut on_fire_count = 0_usize;
    let mut no_physics_count = 0_usize;
    let mut pushout_tick_count = 0_usize;
    let mut removal_tick = None;
    let mut removal_reason = String::new();
    let mut first_on_fire_tick = None;
    let mut first_in_water_tick = None;
    let mut first_underwater_tick = None;
    let mut first_in_lava_tick = None;
    let mut first_underlava_tick = None;

    for row in rows.iter().skip(1) {
        peak_horizontal_speed = peak_horizontal_speed.max(row.horizontal_speed);
        horizontal_speed_sum += row.horizontal_speed;
        moved_count += usize::from(row.moved);
        on_ground_count += usize::from(row.on_ground);
        in_water_count += usize::from(row.in_water);
        underwater_count += usize::from(row.underwater_movement);
        in_lava_count += usize::from(row.in_lava);
        underlava_count += usize::from(row.underlava_movement);
        alive_count += usize::from(row.alive);
        on_fire_count += usize::from(row.on_fire);
        no_physics_count += usize::from(row.no_physics);
        pushout_tick_count += usize::from(row.pushout_applied);
        if removal_tick.is_none() && !row.alive {
            removal_tick = Some(row.tick);
            removal_reason = row.removed_by.clone();
        }
        if first_on_fire_tick.is_none() && row.on_fire {
            first_on_fire_tick = Some(row.tick);
        }
        if first_in_water_tick.is_none() && row.in_water {
            first_in_water_tick = Some(row.tick);
        }
        if first_underwater_tick.is_none() && row.underwater_movement {
            first_underwater_tick = Some(row.tick);
        }
        if first_in_lava_tick.is_none() && row.in_lava {
            first_in_lava_tick = Some(row.tick);
        }
        if first_underlava_tick.is_none() && row.underlava_movement {
            first_underlava_tick = Some(row.tick);
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
        in_lava_ratio: in_lava_count as f64 / tick_denominator,
        underlava_ratio: underlava_count as f64 / tick_denominator,
        alive_ratio: alive_count as f64 / tick_denominator,
        on_fire_ratio: on_fire_count as f64 / tick_denominator,
        removal_tick,
        removal_reason,
        first_on_fire_tick,
        first_in_water_tick,
        first_underwater_tick,
        first_in_lava_tick,
        first_underlava_tick,
        unsupported_collision_block_count: world.approximate_collision_blocks.len(),
        unsupported_collision_blocks: world.approximate_collision_blocks.clone(),
        no_physics_ratio: no_physics_count as f64 / tick_denominator,
        pushout_tick_count,
        collision_model: VERIFY_COLLISION_MODEL,
        fluid_model: VERIFY_FLUID_MODEL,
        fluid_schedule_mode: if command.bootstrap_fluids {
            "bootstrap"
        } else {
            "static_snapshot"
        },
    }
}

fn write_tick_csv(path: &Path, rows: &[VerifyTickRow]) -> Result<(), String> {
    let columns = [
        "tick",
        "alive",
        "removedBy",
        "onFire",
        "remainingFireTicks",
        "health",
        "itemHealth",
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
        "pushoutSpeed",
        "inWater",
        "underwaterMovement",
        "inLava",
        "underlavaMovement",
        "fluidHeight",
        "waterHeight",
        "lavaHeight",
        "activeFluid",
        "activeFluidHeight",
        "fallDistance",
        "currentSamples",
        "appliedCurrentX",
        "appliedCurrentY",
        "appliedCurrentZ",
        "waterCurrentSamples",
        "lavaCurrentSamples",
        "appliedWaterCurrentX",
        "appliedWaterCurrentY",
        "appliedWaterCurrentZ",
        "appliedLavaCurrentX",
        "appliedLavaCurrentY",
        "appliedLavaCurrentZ",
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
        "inLavaRatio",
        "underlavaRatio",
        "aliveRatio",
        "onFireRatio",
        "removalTick",
        "removalReason",
        "firstOnFireTick",
        "firstInWaterTick",
        "firstUnderwaterTick",
        "firstInLavaTick",
        "firstUnderlavaTick",
        "unsupportedCollisionBlockCount",
        "unsupportedCollisionBlocks",
        "noPhysicsRatio",
        "pushoutTickCount",
        "collisionModel",
        "fluidModel",
        "fluidScheduleMode",
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
        metrics.in_lava_ratio.to_string(),
        metrics.underlava_ratio.to_string(),
        metrics.alive_ratio.to_string(),
        metrics.on_fire_ratio.to_string(),
        option_usize(metrics.removal_tick),
        metrics.removal_reason.clone(),
        option_usize(metrics.first_on_fire_tick),
        option_usize(metrics.first_in_water_tick),
        option_usize(metrics.first_underwater_tick),
        option_usize(metrics.first_in_lava_tick),
        option_usize(metrics.first_underlava_tick),
        metrics.unsupported_collision_block_count.to_string(),
        metrics.unsupported_collision_blocks.join(";"),
        metrics.no_physics_ratio.to_string(),
        metrics.pushout_tick_count.to_string(),
        metrics.collision_model.to_string(),
        metrics.fluid_model.to_string(),
        metrics.fluid_schedule_mode.to_string(),
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
            lava_flow_scale: LAVA_FLOW_SCALE,
            horizontal_water_damping: HORIZONTAL_WATER_DAMPING,
            horizontal_lava_damping: HORIZONTAL_LAVA_DAMPING,
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
            no_physics_pushout_speed_min: NO_PHYSICS_PUSHOUT_SPEED_MIN,
            no_physics_pushout_speed_max: NO_PHYSICS_PUSHOUT_SPEED_MAX,
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
        "alive" => row.alive.to_string(),
        "removedBy" => row.removed_by.clone(),
        "onFire" => row.on_fire.to_string(),
        "remainingFireTicks" => row.remaining_fire_ticks.to_string(),
        "health" => row
            .health
            .map(|value| value.to_string())
            .unwrap_or_default(),
        "itemHealth" => row
            .item_health
            .map(|value| value.to_string())
            .unwrap_or_default(),
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
        "pushoutSpeed" => row.pushout_speed.to_string(),
        "inWater" => row.in_water.to_string(),
        "underwaterMovement" => row.underwater_movement.to_string(),
        "inLava" => row.in_lava.to_string(),
        "underlavaMovement" => row.underlava_movement.to_string(),
        "fluidHeight" => row.fluid_height.to_string(),
        "waterHeight" => row.water_height.to_string(),
        "lavaHeight" => row.lava_height.to_string(),
        "activeFluid" => row.active_fluid.clone(),
        "activeFluidHeight" => row.active_fluid_height.to_string(),
        "fallDistance" => row.fall_distance.to_string(),
        "currentSamples" => row.current_samples.to_string(),
        "appliedCurrentX" => row.applied_current_x.to_string(),
        "appliedCurrentY" => row.applied_current_y.to_string(),
        "appliedCurrentZ" => row.applied_current_z.to_string(),
        "waterCurrentSamples" => row.water_current_samples.to_string(),
        "lavaCurrentSamples" => row.lava_current_samples.to_string(),
        "appliedWaterCurrentX" => row.applied_water_current_x.to_string(),
        "appliedWaterCurrentY" => row.applied_water_current_y.to_string(),
        "appliedWaterCurrentZ" => row.applied_water_current_z.to_string(),
        "appliedLavaCurrentX" => row.applied_lava_current_x.to_string(),
        "appliedLavaCurrentY" => row.applied_lava_current_y.to_string(),
        "appliedLavaCurrentZ" => row.applied_lava_current_z.to_string(),
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

fn centered_block_pos(pos: Vec3d, height: f64) -> [i32; 3] {
    [
        pos.x.floor() as i32,
        (pos.y + height * 0.5).floor() as i32,
        pos.z.floor() as i32,
    ]
}

fn centered_block(region: &Region, pos: Vec3d, height: f64) -> Option<&Block> {
    block_at(region, centered_block_pos(pos, height))
}

fn scaffolding_bottom_collision_applies(
    block: &Block,
    block_pos: [i32; 3],
    entity_bottom: f64,
    descending: bool,
) -> bool {
    if descending || block.namespace != "minecraft" || block.id != "scaffolding" {
        return false;
    }
    let distance = block
        .attributes
        .get("distance")
        .and_then(|value| value.parse::<u8>().ok())
        .unwrap_or(7);
    let bottom = block
        .attributes
        .get("bottom")
        .map(|value| value == "true")
        .unwrap_or(false);
    distance != 0 && bottom && entity_bottom > block_pos[1] as f64 - 1.0e-5
}

fn entity_is_above_scaffolding_top(
    block_pos: [i32; 3],
    entity_bottom: f64,
    descending: bool,
) -> bool {
    !descending && entity_bottom > block_pos[1] as f64 + 1.0 - 1.0e-5
}

fn is_climbable_block(block: &Block) -> bool {
    block.namespace == "minecraft"
        && matches!(
            block.id.as_str(),
            "ladder"
                | "vine"
                | "scaffolding"
                | "weeping_vines"
                | "weeping_vines_plant"
                | "twisting_vines"
                | "twisting_vines_plant"
                | "cave_vines"
                | "cave_vines_plant"
        )
}

fn trapdoor_usable_as_ladder(region: &Region, pos: [i32; 3], block: &Block) -> bool {
    if block.namespace != "minecraft"
        || !block.id.ends_with("trapdoor")
        || !bool_attr(block, "open")
    {
        return false;
    }
    let Some(trapdoor_facing) = facing_attr(block) else {
        return false;
    };
    let below_pos = [pos[0], pos[1] - 1, pos[2]];
    let Some(below_block) = block_at(region, below_pos) else {
        return false;
    };
    below_block.namespace == "minecraft"
        && below_block.id == "ladder"
        && facing_attr(below_block) == Some(trapdoor_facing)
}

fn living_block_pos(pos: Vec3d) -> [i32; 3] {
    [
        pos.x.floor() as i32,
        pos.y.floor() as i32,
        pos.z.floor() as i32,
    ]
}

fn living_on_climbable(region: &Region, pos: Vec3d) -> bool {
    let block_pos = living_block_pos(pos);
    let Some(block) = block_at(region, block_pos) else {
        return false;
    };
    is_climbable_block(block) || trapdoor_usable_as_ladder(region, block_pos, block)
}

fn apply_living_climbable_motion(
    region: &Region,
    pos: Vec3d,
    vel: &mut Vec3d,
    fall_distance: &mut f64,
) -> bool {
    if !living_on_climbable(region, pos) {
        return false;
    }
    *fall_distance = 0.0;
    vel.x = vel
        .x
        .clamp(-LIVING_CLIMBABLE_MAX_DELTA, LIVING_CLIMBABLE_MAX_DELTA);
    vel.y = vel.y.max(-LIVING_CLIMBABLE_MAX_DELTA);
    vel.z = vel
        .z
        .clamp(-LIVING_CLIMBABLE_MAX_DELTA, LIVING_CLIMBABLE_MAX_DELTA);
    true
}

fn maybe_jump_living_out_of_fluid(
    region: &Region,
    pos: Vec3d,
    from_y: f64,
    width: f64,
    height: f64,
    fall_distance: f64,
    vel: &mut Vec3d,
    horizontal_collision: bool,
) {
    if !horizontal_collision {
        return;
    }
    let jump_check = Vec3d::new(
        vel.x,
        vel.y + LIVING_FLUID_JUMP_OUT_CLEARANCE - pos.y + from_y,
        vel.z,
    );
    let jump_box = entity_aabb(pos.add(jump_check), width, height);
    if !aabb_intersects_world(region, jump_box, fall_distance, VerifyEntityKind::Living)
        && !aabb_contains_any_fluid(region, jump_box)
    {
        vel.y = LIVING_FLUID_JUMP_OUT_VELOCITY;
    }
}

fn is_climbable_for_fall_reset(block: &Block) -> bool {
    is_climbable_block(block)
}

fn is_fall_damage_resetting_block(block: &Block) -> bool {
    block.namespace == "minecraft" && matches!(block.id.as_str(), "sweet_berry_bush" | "cobweb")
        || is_climbable_for_fall_reset(block)
}

fn movement_resets_fall_distance(region: &Region, from_pos: Vec3d, movement: Vec3d) -> bool {
    let movement_length = movement.length();
    if movement_length < 1.0 {
        return false;
    }
    let check_to = from_pos.add(movement.normalized().scale(movement_length.min(8.0)));
    segment_hits_fall_damage_reset(region, from_pos, check_to)
}

fn segment_hits_fall_damage_reset(region: &Region, from: Vec3d, to: Vec3d) -> bool {
    let min_x = from.x.min(to.x).floor() as i32 - 1;
    let max_x = from.x.max(to.x).floor() as i32 + 1;
    let min_y = from.y.min(to.y).floor() as i32 - 1;
    let max_y = from.y.max(to.y).floor() as i32 + 1;
    let min_z = from.z.min(to.z).floor() as i32 - 1;
    let max_z = from.z.max(to.z).floor() as i32 + 1;

    for x in min_x..=max_x {
        for y in min_y..=max_y {
            for z in min_z..=max_z {
                let block_pos = [x, y, z];
                if let Some(block) = block_at(region, block_pos) {
                    if is_fall_damage_resetting_block(block)
                        && segment_intersects_aabb(from, to, full_block_aabb(block_pos))
                    {
                        return true;
                    }
                }
                let Some(water) = water_at(region, block_pos) else {
                    continue;
                };
                let water_box = Aabb {
                    min_x: x as f64,
                    min_y: y as f64,
                    min_z: z as f64,
                    max_x: x as f64 + 1.0,
                    max_y: y as f64 + water.height,
                    max_z: z as f64 + 1.0,
                };
                if segment_intersects_aabb(from, to, water_box) {
                    return true;
                }
            }
        }
    }

    false
}

fn segment_intersects_aabb(from: Vec3d, to: Vec3d, aabb: Aabb) -> bool {
    let direction = Vec3d::new(to.x - from.x, to.y - from.y, to.z - from.z);
    let mut t_min: f64 = 0.0;
    let mut t_max: f64 = 1.0;

    for (start, delta, min, max) in [
        (from.x, direction.x, aabb.min_x, aabb.max_x),
        (from.y, direction.y, aabb.min_y, aabb.max_y),
        (from.z, direction.z, aabb.min_z, aabb.max_z),
    ] {
        if delta.abs() <= 1.0e-12 {
            if start < min || start > max {
                return false;
            }
            continue;
        }

        let inv_delta = 1.0 / delta;
        let mut t0 = (min - start) * inv_delta;
        let mut t1 = (max - start) * inv_delta;
        if t0 > t1 {
            std::mem::swap(&mut t0, &mut t1);
        }
        t_min = t_min.max(t0);
        t_max = t_max.min(t1);
        if t_max < t_min {
            return false;
        }
    }

    true
}

fn block_pos_below_that_affects_movement(pos: Vec3d, entity_kind: VerifyEntityKind) -> [i32; 3] {
    [
        pos.x.floor() as i32,
        (pos.y - movement_support_offset(entity_kind)).floor() as i32,
        pos.z.floor() as i32,
    ]
}

fn support_block_for(region: &Region, pos: Vec3d, entity_kind: VerifyEntityKind) -> Option<&Block> {
    block_at(
        region,
        block_pos_below_that_affects_movement(pos, entity_kind),
    )
}

fn on_pos_legacy_block(region: &Region, pos: Vec3d) -> Option<&Block> {
    block_at(region, on_pos_legacy_block_pos(pos))
}

fn entity_has_ground_support(
    region: &Region,
    pos: Vec3d,
    width: f64,
    height: f64,
    fall_distance: f64,
    entity_kind: VerifyEntityKind,
) -> bool {
    move_entity_with_fall_distance(
        region,
        pos,
        Vec3d::new(0.0, -GROUND_SUPPORT_PROBE_DELTA, 0.0),
        width,
        height,
        fall_distance,
        entity_kind,
    )
    .collided_y
}

fn block_at_entity_position(region: &Region, pos: Vec3d) -> Option<&Block> {
    block_at(
        region,
        [
            pos.x.floor() as i32,
            pos.y.floor() as i32,
            pos.z.floor() as i32,
        ],
    )
}

fn block_speed_factor_of(block: &Block) -> f64 {
    if block.namespace == "minecraft" {
        match block.id.as_str() {
            "soul_sand" | "honey_block" => 0.4_f32 as f64,
            _ => 1.0,
        }
    } else {
        1.0
    }
}

fn maybe_trigger_big_dripleaf(
    region: &mut Region,
    block_ticks: &mut DynamicBlockTicks,
    pos: Vec3d,
    width: f64,
    on_ground: bool,
    tick: usize,
) {
    if !on_ground {
        return;
    }

    let entity_box = entity_aabb(pos, width, 0.0);
    let min_x = (entity_box.min_x + 1.0e-7).floor() as i32;
    let max_x = (entity_box.max_x - 1.0e-7).floor() as i32;
    let min_z = (entity_box.min_z + 1.0e-7).floor() as i32;
    let max_z = (entity_box.max_z - 1.0e-7).floor() as i32;
    let block_y = (pos.y - 1.0e-7).floor() as i32;

    for x in min_x..=max_x {
        for z in min_z..=max_z {
            let block_pos = [x, block_y, z];
            let Some(block) = block_at(region, block_pos).cloned() else {
                continue;
            };
            if block.namespace != "minecraft" || block.id != "big_dripleaf" {
                continue;
            }
            if big_dripleaf_tilt(&block) != "none"
                || pos.y <= block_pos[1] as f64 + 11.0 / 16.0
                || block_has_neighbor_signal(region, block_pos)
            {
                continue;
            }
            let _ = region.set_block(block_pos, &with_attr(&block, "tilt", "unstable"));
            block_ticks.schedule_big_dripleaf_tick_if_needed(region, tick, block_pos);
        }
    }
}

fn block_speed_factor_for(region: &Region, pos: Vec3d, entity_kind: VerifyEntityKind) -> f64 {
    let state_here = block_at_entity_position(region, pos);
    let speed_factor_here = state_here.map(block_speed_factor_of).unwrap_or(1.0);
    if let Some(block) = state_here {
        if block.namespace == "minecraft" && matches!(block.id.as_str(), "water" | "bubble_column")
        {
            return speed_factor_here;
        }
        if (speed_factor_here - 1.0).abs() > 1.0e-12 {
            return speed_factor_here;
        }
    }
    support_block_for(region, pos, entity_kind)
        .map(block_speed_factor_of)
        .unwrap_or(1.0)
}

fn ground_profile_at_for(
    region: &Region,
    pos: Vec3d,
    entity_kind: VerifyEntityKind,
) -> GroundProfile {
    let Some(block) = support_block_for(region, pos, entity_kind) else {
        return GroundProfile {
            friction: 0.6_f32 as f64,
            slime_surface: false,
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
    let slime_surface = on_pos_legacy_block(region, pos)
        .map(|surface| surface.namespace == "minecraft" && surface.id == "slime_block")
        .unwrap_or(false);
    GroundProfile {
        friction,
        slime_surface,
    }
}

fn horizontal_drag(profile: GroundProfile, on_ground: bool, vy: f64) -> f64 {
    let mut drag = HORIZONTAL_MOVEMENT_DAMPING;
    if on_ground {
        drag = profile.friction * HORIZONTAL_MOVEMENT_DAMPING;
        if profile.slime_surface && vy.abs() < SLIME_STEP_ON_VY_THRESHOLD {
            drag *= SLIME_STEP_ON_BASE + SLIME_STEP_ON_VY_SCALE * vy.abs();
        }
    }
    drag
}

fn vertical_velocity_after_landing(
    on_slime_surface: bool,
    current_velocity_y: f64,
    actual_movement_y: f64,
    gravity: f64,
    entity_kind: VerifyEntityKind,
) -> f64 {
    if !on_slime_surface || current_velocity_y >= 0.0 || -current_velocity_y < gravity {
        return 0.0;
    }

    let restitution = if matches!(entity_kind, VerifyEntityKind::Living) {
        1.0
    } else {
        0.8
    };
    let movement_portion = actual_movement_y / current_velocity_y;
    let gravity_compensation = movement_portion * gravity;
    let effective_drag = 1.0 + movement_portion * (VERTICAL_MOVEMENT_DAMPING - 1.0);
    (gravity_compensation - current_velocity_y) * effective_drag * restitution
}

fn track_fluid(
    region: &Region,
    pos: Vec3d,
    width: f64,
    height: f64,
    fluid_kind: FluidKind,
    ignore_current: bool,
) -> FluidTracker {
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
    let mut tracker = FluidTracker::default();

    for x in x0..=x1 {
        for y in y0..=y1 {
            for z in z0..=z1 {
                let cell_pos = [x, y, z];
                let fluid_cell = match fluid_kind {
                    FluidKind::Water => water_at(region, cell_pos),
                    FluidKind::Lava => lava_at(region, cell_pos),
                };
                let Some(fluid_cell) = fluid_cell else {
                    continue;
                };
                let fluid_top = y as f64 + fluid_cell.height;
                if fluid_top < box_min_y {
                    continue;
                }
                tracker.height = tracker.height.max(fluid_top - pos.y);
                if !ignore_current {
                    let mut flow = fluid_flow(region, cell_pos, fluid_cell, fluid_kind);
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

fn track_water(
    region: &Region,
    pos: Vec3d,
    width: f64,
    height: f64,
    ignore_current: bool,
) -> FluidTracker {
    track_fluid(region, pos, width, height, FluidKind::Water, ignore_current)
}

fn track_lava(
    region: &Region,
    pos: Vec3d,
    width: f64,
    height: f64,
    ignore_current: bool,
) -> FluidTracker {
    track_fluid(region, pos, width, height, FluidKind::Lava, ignore_current)
}

fn fluid_flow(
    region: &Region,
    pos: [i32; 3],
    fluid_cell: WaterCell,
    fluid_kind: FluidKind,
) -> Vec3d {
    let mut flow_x = 0.0;
    let mut flow_z = 0.0;
    for (dx, dz) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
        let neighbor_pos = [pos[0] + dx, pos[1], pos[2] + dz];
        let neighbor_fluid = fluid_at(region, neighbor_pos, fluid_kind);
        let mut neighbor_height = neighbor_fluid.map(|cell| cell.own_height).unwrap_or(0.0);
        let mut distance = 0.0;
        if neighbor_height == 0.0 {
            let neighbor_block = block_at(region, neighbor_pos);
            if neighbor_block
                .map(|block| !fluid_blocks_motion(block))
                .unwrap_or(true)
            {
                let below_neighbor = fluid_at(
                    region,
                    [neighbor_pos[0], neighbor_pos[1] - 1, neighbor_pos[2]],
                    fluid_kind,
                );
                if let Some(below_neighbor) = below_neighbor {
                    neighbor_height = below_neighbor.own_height;
                    if neighbor_height > 0.0 {
                        distance =
                            fluid_cell.own_height - (neighbor_height - 0.888_888_9_f32 as f64);
                    }
                }
            }
        } else {
            distance = fluid_cell.own_height - neighbor_height;
        }
        if distance != 0.0 {
            flow_x += dx as f64 * distance;
            flow_z += dz as f64 * distance;
        }
    }

    let mut flow = Vec3d::new(flow_x, 0.0, flow_z).normalized();
    if fluid_cell.falling {
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

fn apply_fluid_current(velocity: &mut Vec3d, tracker: FluidTracker, scale: f64) -> Vec3d {
    if tracker.current_count == 0
        || tracker.accumulated_current.length_sqr() < FLUID_CURRENT_EPSILON2
    {
        return Vec3d::ZERO;
    }

    let mut impulse = tracker.accumulated_current.normalized().scale(scale);
    if velocity.x.abs() < FLUID_CURRENT_MIN_OLD_MOVEMENT
        && velocity.z.abs() < FLUID_CURRENT_MIN_OLD_MOVEMENT
        && impulse.length() < FLUID_CURRENT_MIN_IMPULSE
    {
        impulse = impulse.normalized().scale(FLUID_CURRENT_MIN_IMPULSE);
    }
    *velocity = velocity.add(impulse);
    impulse
}

fn apply_water_current(velocity: &mut Vec3d, tracker: FluidTracker) -> Vec3d {
    apply_fluid_current(velocity, tracker, WATER_PUSH)
}

fn apply_lava_current(velocity: &mut Vec3d, tracker: FluidTracker) -> Vec3d {
    apply_fluid_current(velocity, tracker, LAVA_FLOW_SCALE)
}

#[cfg(test)]
fn move_entity(
    region: &Region,
    pos: Vec3d,
    velocity: Vec3d,
    width: f64,
    height: f64,
) -> MoveResult {
    move_entity_with_fall_distance(
        region,
        pos,
        velocity,
        width,
        height,
        0.0,
        VerifyEntityKind::Item,
    )
}

fn move_entity_with_fall_distance(
    region: &Region,
    pos: Vec3d,
    velocity: Vec3d,
    width: f64,
    height: f64,
    fall_distance: f64,
    entity_kind: VerifyEntityKind,
) -> MoveResult {
    let mut aabb = entity_aabb(pos, width, height);
    let (dy, collided_y) = collide_axis(
        region,
        aabb,
        velocity.y,
        Axis::Y,
        fall_distance,
        entity_kind,
    );
    aabb = shift_aabb(aabb, Axis::Y, dy);

    let mut dx = 0.0;
    let mut dz = 0.0;
    let mut collided_x = false;
    let mut collided_z = false;
    for axis in horizontal_axes_in_order(velocity) {
        let (resolved, collided) = collide_axis(
            region,
            aabb,
            axis_component(velocity, axis),
            axis,
            fall_distance,
            entity_kind,
        );
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

fn is_no_physics(
    region: &Region,
    pos: Vec3d,
    width: f64,
    height: f64,
    fall_distance: f64,
    entity_kind: VerifyEntityKind,
) -> bool {
    if !matches!(entity_kind, VerifyEntityKind::Item) {
        return false;
    }
    aabb_intersects_world(
        region,
        deflate_aabb(entity_aabb(pos, width, height), NO_PHYSICS_DEFLATE),
        fall_distance,
        entity_kind,
    )
}

fn move_towards_closest_space(
    region: &Region,
    pos: Vec3d,
    velocity: Vec3d,
    height: f64,
    rng: Option<&mut LegacyRandom>,
) -> (Vec3d, f64) {
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

    let speed = rng
        .map(|rng| NO_PHYSICS_PUSHOUT_SPEED_MIN + rng.next_float() * 0.2)
        .unwrap_or(NO_PHYSICS_PUSHOUT_SPEED);
    let scaled = velocity.scale(0.75);
    let adjusted = match closest_axis {
        Axis::X => Vec3d::new(closest_step * speed, scaled.y, scaled.z),
        Axis::Y => Vec3d::new(scaled.x, closest_step * speed, scaled.z),
        Axis::Z => Vec3d::new(scaled.x, scaled.y, closest_step * speed),
    };
    (adjusted, speed)
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

fn aabb_intersects_world(
    region: &Region,
    aabb: Aabb,
    fall_distance: f64,
    entity_kind: VerifyEntityKind,
) -> bool {
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
                if for_entity_collision_box(
                    block,
                    [x, y, z],
                    fall_distance,
                    entity_kind,
                    aabb.min_y,
                    false,
                    |collision_box| {
                        if aabbs_intersect(
                            aabb,
                            world_collision_box(block, [x, y, z], collision_box),
                        ) {
                            return Some(());
                        }
                        None
                    },
                )
                .is_some()
                {
                    return true;
                }
            }
        }
    }
    false
}

fn aabb_contains_any_fluid(region: &Region, aabb: Aabb) -> bool {
    let x0 = aabb.min_x.floor() as i32;
    let x1 = aabb.max_x.ceil() as i32 - 1;
    let y0 = aabb.min_y.floor() as i32;
    let y1 = aabb.max_y.ceil() as i32 - 1;
    let z0 = aabb.min_z.floor() as i32;
    let z1 = aabb.max_z.ceil() as i32 - 1;

    for x in x0..=x1 {
        for y in y0..=y1 {
            for z in z0..=z1 {
                let block_pos = [x, y, z];
                for fluid_kind in [FluidKind::Water, FluidKind::Lava] {
                    let Some(fluid) = fluid_at(region, block_pos, fluid_kind) else {
                        continue;
                    };
                    let fluid_top = block_pos[1] as f64 + fluid.height;
                    if fluid_top > aabb.min_y + 1.0e-12
                        && (block_pos[0] as f64 + 1.0) > aabb.min_x + 1.0e-12
                        && (block_pos[0] as f64) < aabb.max_x - 1.0e-12
                        && (block_pos[2] as f64 + 1.0) > aabb.min_z + 1.0e-12
                        && (block_pos[2] as f64) < aabb.max_z - 1.0e-12
                        && (block_pos[1] as f64) < aabb.max_y - 1.0e-12
                    {
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

fn moving_aabb_intersects_aabb(moving: Aabb, delta: Vec3d, target: Aabb) -> bool {
    let mut entry_time = 0.0_f64;
    let mut exit_time = 1.0_f64;

    for (moving_min, moving_max, target_min, target_max, axis_delta) in [
        (
            moving.min_x,
            moving.max_x,
            target.min_x,
            target.max_x,
            delta.x,
        ),
        (
            moving.min_y,
            moving.max_y,
            target.min_y,
            target.max_y,
            delta.y,
        ),
        (
            moving.min_z,
            moving.max_z,
            target.min_z,
            target.max_z,
            delta.z,
        ),
    ] {
        if axis_delta.abs() <= 1.0e-12 {
            if moving_max <= target_min + 1.0e-12 || moving_min >= target_max - 1.0e-12 {
                return false;
            }
            continue;
        }

        let (axis_entry, axis_exit) = if axis_delta > 0.0 {
            (
                (target_min - moving_max) / axis_delta,
                (target_max - moving_min) / axis_delta,
            )
        } else {
            (
                (target_max - moving_min) / axis_delta,
                (target_min - moving_max) / axis_delta,
            )
        };

        entry_time = entry_time.max(axis_entry.min(axis_exit));
        exit_time = exit_time.min(axis_entry.max(axis_exit));
        if entry_time > exit_time {
            return false;
        }
    }

    exit_time >= 0.0 && entry_time <= 1.0
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

fn guardian_block_seed(x: i32, y: i32, z: i32) -> i64 {
    let mut seed =
        (x as i64).wrapping_mul(3_129_871) ^ (z as i64).wrapping_mul(116_129_781) ^ y as i64;
    seed = seed
        .wrapping_mul(seed)
        .wrapping_mul(42_317_861)
        .wrapping_add(seed.wrapping_mul(11));
    seed >> 16
}

fn vanilla_horizontal_block_offset(seed: u64, shift: u32, max_horizontal_offset: f32) -> f64 {
    let nibble = ((seed >> shift) & 15) as f32;
    let offset = ((nibble / 15.0_f32) - 0.5_f32) * 0.5_f32;
    offset.clamp(-max_horizontal_offset, max_horizontal_offset) as f64
}

fn block_collision_offset(block: &Block, pos: [i32; 3]) -> Vec3d {
    if block.namespace != "minecraft" {
        return Vec3d::ZERO;
    }

    let max_horizontal_offset = match block.id.as_str() {
        "pointed_dripstone" => 0.125_f32,
        "bamboo" => 0.25_f32,
        _ => return Vec3d::ZERO,
    };
    let seed = guardian_block_seed(pos[0], 0, pos[2]) as u64;
    let x = vanilla_horizontal_block_offset(seed, 0, max_horizontal_offset);
    let z = vanilla_horizontal_block_offset(seed, 8, max_horizontal_offset);
    Vec3d::new(x, 0.0, z)
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

fn world_collision_box(block: &Block, block_pos: [i32; 3], collision_box: &CollisionBox) -> Aabb {
    let offset = block_collision_offset(block, block_pos);
    Aabb {
        min_x: block_pos[0] as f64 + collision_box.min_x + offset.x,
        min_y: block_pos[1] as f64 + collision_box.min_y,
        min_z: block_pos[2] as f64 + collision_box.min_z + offset.z,
        max_x: block_pos[0] as f64 + collision_box.max_x + offset.x,
        max_y: block_pos[1] as f64 + collision_box.max_y,
        max_z: block_pos[2] as f64 + collision_box.max_z + offset.z,
    }
}

fn for_entity_collision_box<T>(
    block: &Block,
    block_pos: [i32; 3],
    fall_distance: f64,
    entity_kind: VerifyEntityKind,
    entity_bottom: f64,
    descending: bool,
    mut f: impl FnMut(&CollisionBox) -> Option<T>,
) -> Option<T> {
    if block.namespace == "minecraft" && block.id == "powder_snow" {
        if fall_distance > POWDER_SNOW_FALL_DISTANCE_COLLISION_THRESHOLD {
            let collision_box = CollisionBox {
                min_x: 0.0,
                min_y: 0.0,
                min_z: 0.0,
                max_x: 1.0,
                max_y: 0.9_f32 as f64,
                max_z: 1.0,
            };
            return f(&collision_box);
        }
        if powder_snow_has_walkable_collision(entity_kind) {
            let collision_box = CollisionBox {
                min_x: 0.0,
                min_y: 0.0,
                min_z: 0.0,
                max_x: 1.0,
                max_y: 1.0,
                max_z: 1.0,
            };
            return f(&collision_box);
        }
        return None;
    }

    if block.namespace == "minecraft" && block.id == "scaffolding" {
        if entity_is_above_scaffolding_top(block_pos, entity_bottom, descending) {
            for collision_box in collision_boxes(block).iter() {
                if let Some(value) = f(collision_box) {
                    return Some(value);
                }
            }
            return None;
        }
        if scaffolding_bottom_collision_applies(block, block_pos, entity_bottom, descending) {
            let collision_box = CollisionBox {
                min_x: 0.0,
                min_y: 0.0,
                min_z: 0.0,
                max_x: 1.0,
                max_y: 2.0 / 16.0,
                max_z: 1.0,
            };
            return f(&collision_box);
        }
        return None;
    }

    for collision_box in collision_boxes(block).iter() {
        if let Some(value) = f(collision_box) {
            return Some(value);
        }
    }
    None
}

fn collide_axis(
    region: &Region,
    aabb: Aabb,
    delta: f64,
    axis: Axis,
    fall_distance: f64,
    entity_kind: VerifyEntityKind,
) -> (f64, bool) {
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
                let _ = for_entity_collision_box(
                    block,
                    [x, y, z],
                    fall_distance,
                    entity_kind,
                    aabb.min_y,
                    false,
                    |collision_box| {
                        let world_box = world_collision_box(block, [x, y, z], collision_box);
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
                                    return None;
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
                                    return None;
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
                                    return None;
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
                        None::<()>
                    },
                );
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
    use crate::litematic::{CollisionKind, collision_boxes, collision_kind};
    use crate::{VERIFY_ARMOR_STAND_HEIGHT, VERIFY_ARMOR_STAND_WIDTH};

    fn parse_block(id: &str) -> Block {
        Block::from_id(id).expect("valid block")
    }

    fn region_with_shape(shape: [i32; 3]) -> Region {
        Region::with_shape(shape)
    }

    fn armor_stand_verify_command(input: &str) -> VerifyCommand {
        VerifyCommand {
            input: std::path::PathBuf::from(input),
            out: std::path::PathBuf::from("artifacts/test"),
            target_speed: 0.0,
            ticks: 1,
            inspect_tick: Some(1),
            start_x: 0.0,
            start_y: 0.0,
            start_z: 0.0,
            start_vx: 0.0,
            start_vy: 0.0,
            start_vz: 0.0,
            start_on_ground: false,
            width: VERIFY_ARMOR_STAND_WIDTH,
            height: VERIFY_ARMOR_STAND_HEIGHT,
            entity_id_mod4: 0,
            initial_tick_count: 0,
            entity_rng_seed: None,
            entity_uuid: None,
            bootstrap_fluids: false,
            entity_kind: VerifyEntityKind::Living,
            no_ai: false,
            no_gravity: false,
            fire_immune: false,
            start_fire_ticks: 0,
            item_health: Some(10),
        }
    }

    fn armor_stand_magma_probe_world() -> LoadedSchematic {
        let mut region = region_with_shape([3, 67, 3]);
        for x in 0..=2 {
            for z in 0..=2 {
                region
                    .set_block([x, 63, z], &parse_block("minecraft:magma_block"))
                    .unwrap();
            }
        }
        LoadedSchematic {
            name: "armor-stand-magma-probe".to_string(),
            region,
            approximate_collision_blocks: Vec::new(),
        }
    }

    fn armor_stand_pointed_dripstone_probe_world() -> LoadedSchematic {
        let mut region = region_with_shape([3, 67, 3]);
        region
            .set_block(
                [1, 63, 1],
                &parse_block(
                    "minecraft:pointed_dripstone[thickness=tip,vertical_direction=up,waterlogged=false]",
                ),
            )
            .unwrap();
        LoadedSchematic {
            name: "armor-stand-pointed-dripstone-probe".to_string(),
            region,
            approximate_collision_blocks: Vec::new(),
        }
    }

    fn armor_stand_ladder_probe_world() -> LoadedSchematic {
        let mut region = region_with_shape([3, 69, 3]);
        region
            .set_block([1, 65, 1], &parse_block("minecraft:ladder[facing=north]"))
            .unwrap();
        LoadedSchematic {
            name: "armor-stand-ladder-probe".to_string(),
            region,
            approximate_collision_blocks: Vec::new(),
        }
    }

    fn armor_stand_trapdoor_ladder_probe_world() -> LoadedSchematic {
        let mut region = region_with_shape([3, 69, 3]);
        region
            .set_block([1, 64, 1], &parse_block("minecraft:ladder[facing=north]"))
            .unwrap();
        region
            .set_block(
                [1, 65, 1],
                &parse_block(
                    "minecraft:oak_trapdoor[facing=north,half=bottom,open=true,waterlogged=false]",
                ),
            )
            .unwrap();
        LoadedSchematic {
            name: "armor-stand-trapdoor-ladder-probe".to_string(),
            region,
            approximate_collision_blocks: Vec::new(),
        }
    }

    fn armor_stand_ladder_horizontal_probe_world() -> LoadedSchematic {
        let mut region = region_with_shape([4, 69, 3]);
        region
            .set_block([1, 65, 0], &parse_block("minecraft:stone"))
            .unwrap();
        region
            .set_block([1, 65, 1], &parse_block("minecraft:ladder[facing=north]"))
            .unwrap();
        region
            .set_block([2, 65, 1], &parse_block("minecraft:stone"))
            .unwrap();
        LoadedSchematic {
            name: "armor-stand-ladder-horizontal-probe".to_string(),
            region,
            approximate_collision_blocks: Vec::new(),
        }
    }

    fn armor_stand_water_horizontal_probe_world() -> LoadedSchematic {
        let mut region = region_with_shape([4, 69, 3]);
        region
            .set_block([1, 64, 1], &parse_block("minecraft:water"))
            .unwrap();
        region
            .set_block([2, 64, 1], &parse_block("minecraft:stone"))
            .unwrap();
        LoadedSchematic {
            name: "armor-stand-water-horizontal-probe".to_string(),
            region,
            approximate_collision_blocks: Vec::new(),
        }
    }

    fn armor_stand_water_diagonal_wall_probe_world() -> LoadedSchematic {
        let mut region = region_with_shape([5, 69, 4]);
        region
            .set_block([1, 64, 1], &parse_block("minecraft:water"))
            .unwrap();
        for x in 0..=4 {
            region
                .set_block([x, 64, 2], &parse_block("minecraft:stone"))
                .unwrap();
        }
        LoadedSchematic {
            name: "armor-stand-water-diagonal-wall-probe".to_string(),
            region,
            approximate_collision_blocks: Vec::new(),
        }
    }

    fn item_bamboo_horizontal_probe_world() -> LoadedSchematic {
        let mut region = region_with_shape([5, 69, 3]);
        region
            .set_block([1, 64, 1], &parse_block("minecraft:sand"))
            .unwrap();
        region
            .set_block(
                [1, 65, 1],
                &parse_block("minecraft:bamboo[age=0,leaves=none,stage=0]"),
            )
            .unwrap();
        LoadedSchematic {
            name: "item-bamboo-horizontal-probe".to_string(),
            region,
            approximate_collision_blocks: Vec::new(),
        }
    }

    fn item_unsupported_bamboo_snapshot_world() -> LoadedSchematic {
        let mut region = region_with_shape([5, 69, 3]);
        region
            .set_block(
                [1, 65, 1],
                &parse_block("minecraft:bamboo[age=0,leaves=none,stage=0]"),
            )
            .unwrap();
        LoadedSchematic {
            name: "item-unsupported-bamboo-snapshot".to_string(),
            region,
            approximate_collision_blocks: Vec::new(),
        }
    }

    fn item_unsupported_scaffolding_snapshot_world() -> LoadedSchematic {
        let mut region = region_with_shape([5, 69, 3]);
        region
            .set_block(
                [1, 65, 1],
                &parse_block("minecraft:scaffolding[bottom=false,distance=0,waterlogged=false]"),
            )
            .unwrap();
        LoadedSchematic {
            name: "item-unsupported-scaffolding-snapshot".to_string(),
            region,
            approximate_collision_blocks: Vec::new(),
        }
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

    fn mixed_sign_world() -> LoadedSchematic {
        let mut region = region_with_shape([17, 5, 3]);
        region.name = "mixed_sign".to_string();

        for x in 0..=16 {
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

        for x in 1..=15 {
            region
                .set_block([x, 0, 1], &parse_block("minecraft:packed_ice"))
                .unwrap();
        }

        for (x, block) in [
            (1, "minecraft:water"),
            (2, "minecraft:oak_sign[rotation=0,waterlogged=false]"),
            (3, "minecraft:water[level=2]"),
            (4, "minecraft:oak_wall_sign[facing=north,waterlogged=false]"),
            (5, "minecraft:water[level=4]"),
            (6, "minecraft:oak_fence_gate[facing=north,open=true]"),
            (7, "minecraft:water[level=6]"),
            (
                8,
                "minecraft:iron_bars[north=false,south=false,east=false,west=false,waterlogged=false]",
            ),
            (9, "minecraft:water"),
            (
                10,
                "minecraft:cobblestone_wall[north=none,south=none,east=none,west=none,up=true,waterlogged=false]",
            ),
            (
                11,
                "minecraft:oak_trapdoor[half=bottom,open=true,facing=north,waterlogged=false]",
            ),
            (
                12,
                "minecraft:oak_stairs[facing=east,half=bottom,shape=straight,waterlogged=false]",
            ),
            (13, "minecraft:water[level=1]"),
            (14, "minecraft:oak_fence_gate[facing=north,open=false]"),
            (15, "minecraft:water"),
        ] {
            region.set_block([x, 1, 1], &parse_block(block)).unwrap();
        }

        LoadedSchematic {
            name: "mixed-sign".to_string(),
            region,
            approximate_collision_blocks: Vec::new(),
        }
    }

    fn partial_collision_probe_world(block_id: &str) -> LoadedSchematic {
        let mut region = region_with_shape([5, 4, 2]);
        region.name = "partial_collision_probe".to_string();

        for x in 0..=4 {
            for z in 0..=1 {
                region
                    .set_block([x, 0, z], &parse_block("minecraft:smooth_stone"))
                    .unwrap();
            }
        }

        let block = parse_block(block_id);
        region.set_block([1, 1, 1], &block).unwrap();
        if block.id == "oak_wall_hanging_sign" {
            region
                .set_block([1, 1, 0], &parse_block("minecraft:smooth_stone"))
                .unwrap();
        }

        LoadedSchematic {
            name: format!("partial-collision-{}", block.id),
            region,
            approximate_collision_blocks: Vec::new(),
        }
    }

    fn advanced_collision_probe_world() -> LoadedSchematic {
        let mut region = region_with_shape([15, 3, 10]);
        region.name = "advanced_collision_probe".to_string();

        for x in 0..=14 {
            for z in 0..=9 {
                region
                    .set_block([x, 0, z], &parse_block("minecraft:smooth_stone"))
                    .unwrap();
            }
        }

        region
            .set_block([8, 1, 1], &parse_block("minecraft:smooth_stone"))
            .unwrap();
        region
            .set_block([7, 1, 4], &parse_block("minecraft:smooth_stone"))
            .unwrap();

        for (pos, block) in [
            ([4, 1, 0], "minecraft:white_banner[rotation=0]"),
            ([8, 1, 0], "minecraft:white_wall_banner[facing=north]"),
            ([12, 1, 0], "minecraft:skeleton_skull[rotation=0]"),
            ([4, 1, 4], "minecraft:player_head[rotation=0]"),
            ([8, 1, 4], "minecraft:skeleton_wall_skull[facing=east]"),
            (
                [12, 1, 4],
                "minecraft:decorated_pot[facing=north,cracked=false,waterlogged=false]",
            ),
            (
                [4, 1, 8],
                "minecraft:chest[facing=north,type=single,waterlogged=false]",
            ),
            (
                [8, 1, 8],
                "minecraft:trapped_chest[facing=north,type=single,waterlogged=false]",
            ),
            ([12, 1, 8], "minecraft:barrel[facing=north,open=false]"),
        ] {
            region.set_block(pos, &parse_block(block)).unwrap();
        }

        LoadedSchematic {
            name: "advanced-collision-probe".to_string(),
            region,
            approximate_collision_blocks: Vec::new(),
        }
    }

    fn button_probe_world() -> LoadedSchematic {
        let mut region = region_with_shape([8, 4, 8]);
        region.name = "button_probe".to_string();

        for x in 0..=7 {
            for z in 0..=7 {
                region
                    .set_block([x, 0, z], &parse_block("minecraft:smooth_stone"))
                    .unwrap();
            }
        }

        for (pos, block) in [
            ([4, 1, 1], "minecraft:smooth_stone"),
            ([4, 1, 2], "minecraft:smooth_stone"),
            (
                [5, 1, 1],
                "minecraft:stone_button[face=wall,facing=east,powered=false]",
            ),
            (
                [5, 1, 2],
                "minecraft:stone_button[face=wall,facing=east,powered=true]",
            ),
            (
                [5, 1, 3],
                "minecraft:stone_button[face=floor,facing=north,powered=false]",
            ),
            (
                [5, 1, 4],
                "minecraft:stone_button[face=floor,facing=north,powered=true]",
            ),
            ([5, 2, 5], "minecraft:smooth_stone"),
            ([5, 2, 6], "minecraft:smooth_stone"),
            (
                [5, 1, 5],
                "minecraft:stone_button[face=ceiling,facing=north,powered=false]",
            ),
            (
                [5, 1, 6],
                "minecraft:stone_button[face=ceiling,facing=north,powered=true]",
            ),
        ] {
            region.set_block(pos, &parse_block(block)).unwrap();
        }

        LoadedSchematic {
            name: "button-probe".to_string(),
            region,
            approximate_collision_blocks: Vec::new(),
        }
    }

    fn hopper_probe_world() -> LoadedSchematic {
        let mut region = region_with_shape([9, 4, 8]);
        region.name = "hopper_probe".to_string();

        for x in 0..=8 {
            for z in 0..=7 {
                region
                    .set_block([x, 0, z], &parse_block("minecraft:smooth_stone"))
                    .unwrap();
            }
        }

        for (pos, block) in [
            ([5, 1, 0], "minecraft:hopper[facing=down,enabled=true]"),
            ([5, 1, 2], "minecraft:hopper[facing=east,enabled=true]"),
            ([5, 1, 4], "minecraft:hopper[facing=down,enabled=true]"),
            ([5, 1, 6], "minecraft:hopper[facing=down,enabled=true]"),
        ] {
            region.set_block(pos, &parse_block(block)).unwrap();
            region.block_entities.insert(pos, BlockEntity::new());
        }

        LoadedSchematic {
            name: "hopper-probe".to_string(),
            region,
            approximate_collision_blocks: Vec::new(),
        }
    }

    fn run_fluid_ticks(region: &mut Region, ticks: usize) {
        let mut block_ticks = DynamicBlockTicks::default();
        let mut fluid_ticks = DynamicFluidTicks::bootstrap(region);
        for tick in 1..=ticks {
            fluid_ticks.run_due(region, &mut block_ticks, tick);
        }
    }

    fn run_world_ticks(region: &mut Region, ticks: usize) {
        let mut block_ticks = DynamicBlockTicks::bootstrap(region);
        let mut fluid_ticks = DynamicFluidTicks::bootstrap(region);
        for tick in 1..=ticks {
            block_ticks.run_due(region, tick);
            fluid_ticks.run_due(region, &mut block_ticks, tick);
        }
    }

    fn hopper_container_probe_world() -> LoadedSchematic {
        let mut region = region_with_shape([3, 4, 1]);
        region.name = "hopper_container_probe".to_string();

        for x in 0..=2 {
            region
                .set_block([x, 0, 0], &parse_block("minecraft:smooth_stone"))
                .unwrap();
        }

        region
            .set_block(
                [1, 1, 0],
                &parse_block("minecraft:hopper[facing=down,enabled=true]"),
            )
            .unwrap();
        region
            .set_block(
                [1, 2, 0],
                &parse_block("minecraft:chest[facing=north,type=single,waterlogged=false]"),
            )
            .unwrap();
        region.block_entities.insert([1, 1, 0], BlockEntity::new());
        region.block_entities.insert([1, 2, 0], BlockEntity::new());

        LoadedSchematic {
            name: "hopper-container-probe".to_string(),
            region,
            approximate_collision_blocks: Vec::new(),
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
            FluidKind::Water,
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
            FluidKind::Water,
        );
        assert!(gate_flow.y.abs() < 1.0e-12);

        region
            .set_block([1, 0, 0], &parse_block("minecraft:stone"))
            .unwrap();
        let stone_flow = fluid_flow(
            &region,
            [0, 0, 0],
            water_at(&region, [0, 0, 0]).expect("water"),
            FluidKind::Water,
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
        let (adjusted, speed) = move_towards_closest_space(
            &region,
            Vec3d::new(1.8, 1.0, 1.4),
            Vec3d::new(0.4, 0.2, 0.3),
            0.25,
            None,
        );
        assert!((speed - 0.2).abs() < 1.0e-12);
        assert!((adjusted.x - 0.3).abs() < 1.0e-12);
        assert!((adjusted.y - 0.15).abs() < 1.0e-12);
        assert!((adjusted.z + 0.2).abs() < 1.0e-12);
    }

    #[test]
    fn move_towards_closest_space_uses_seeded_legacy_random_speed() {
        let mut region = region_with_shape([3, 3, 3]);
        for pos in [[1, 1, 1], [0, 1, 1], [2, 1, 1], [1, 1, 2], [1, 2, 1]] {
            region
                .set_block(pos, &parse_block("minecraft:stone"))
                .unwrap();
        }
        let mut rng = LegacyRandom::new(123456789);
        let (adjusted, speed) = move_towards_closest_space(
            &region,
            Vec3d::new(1.8, 1.0, 1.4),
            Vec3d::new(0.4, 0.2, 0.3),
            0.25,
            Some(&mut rng),
        );
        assert!(speed >= NO_PHYSICS_PUSHOUT_SPEED_MIN);
        assert!(speed < NO_PHYSICS_PUSHOUT_SPEED_MAX);
        assert!((adjusted.x - 0.3).abs() < 1.0e-12);
        assert!((adjusted.y - 0.15).abs() < 1.0e-12);
        assert!((adjusted.z + speed).abs() < 1.0e-12);
    }

    #[test]
    fn legacy_random_state_can_be_recovered_from_entity_uuid() {
        let mut original = LegacyRandom::new(123456789);
        let uuid = insecure_uuid_from_legacy_random(&mut original);
        let expected_next_float = original.next_float();

        let recovered_state = legacy_random_state_after_entity_uuid(uuid)
            .expect("uuid should map back to a legacy RNG state");
        let mut recovered = LegacyRandom::from_internal_seed(recovered_state);
        assert!((recovered.next_float() - expected_next_float).abs() < 1.0e-12);
        assert_eq!(
            legacy_random_from_entity_uuid(&uuid.to_string()).map(|rng| rng.seed),
            Some(recovered_state)
        );
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
            entity_rng_seed: None,
            entity_uuid: None,
            bootstrap_fluids: false,
            entity_kind: VerifyEntityKind::Item,
            no_ai: false,

            no_gravity: false,
            fire_immune: false,
            start_fire_ticks: 0,
            item_health: None,
        };

        let rows = simulate(&world, &command);
        assert!(rows[0].no_physics);
        assert!(rows[1].pushout_applied);
        assert!(rows[1].vz < 0.0);
    }

    #[test]
    fn item_support_block_uses_vanilla_movement_offset() {
        let mut region = region_with_shape([1, 3, 1]);
        region
            .set_block([0, 0, 0], &parse_block("minecraft:blue_ice"))
            .unwrap();
        region
            .set_block(
                [0, 1, 0],
                &parse_block("minecraft:oak_slab[type=bottom,waterlogged=false]"),
            )
            .unwrap();

        let support = support_block_for(&region, Vec3d::new(0.5, 1.5, 0.5), VerifyEntityKind::Item)
            .expect("support block");
        assert_eq!(support.id, "blue_ice");
    }

    #[test]
    fn simulate_item_falls_through_flat_rail_like_vanilla_probe() {
        let mut region = region_with_shape([3, 3, 3]);
        region
            .set_block([1, 0, 1], &parse_block("minecraft:stone"))
            .unwrap();
        region
            .set_block(
                [1, 1, 1],
                &parse_block("minecraft:rail[shape=north_south,waterlogged=false]"),
            )
            .unwrap();
        let world = LoadedSchematic {
            name: "flat-rail-item-fallthrough".to_string(),
            region,
            approximate_collision_blocks: Vec::new(),
        };
        let command = VerifyCommand {
            input: std::path::PathBuf::from("flat-rail-item-fallthrough.litematic"),
            out: std::path::PathBuf::from("artifacts/test"),
            target_speed: 0.0,
            ticks: 2,
            inspect_tick: Some(2),
            start_x: 1.5,
            start_y: 1.3,
            start_z: 1.5,
            start_vx: 0.0,
            start_vy: -0.2,
            start_vz: 0.0,
            start_on_ground: false,
            width: VERIFY_DEFAULT_WIDTH,
            height: VERIFY_DEFAULT_HEIGHT,
            entity_id_mod4: 0,
            initial_tick_count: 0,
            entity_rng_seed: None,
            entity_uuid: None,
            bootstrap_fluids: false,
            entity_kind: VerifyEntityKind::Item,
            no_ai: false,
            no_gravity: false,
            fire_immune: false,
            start_fire_ticks: 0,
            item_health: None,
        };

        let rows = simulate(&world, &command);
        let tick1 = &rows[1];
        let tick2 = &rows[2];
        assert!((tick1.x - 1.5).abs() < 5.0e-8);
        assert!((tick1.y - 1.06).abs() < 5.0e-8);
        assert!((tick1.vx - 0.0).abs() < 5.0e-8);
        assert!((tick1.vy - (-0.235_200_004_577_636_73)).abs() < 5.0e-8);
        assert!(!tick1.on_ground);
        assert!((tick2.x - 1.5).abs() < 5.0e-8);
        assert!((tick2.y - 1.0).abs() < 5.0e-8);
        assert!((tick2.vx - 0.0).abs() < 5.0e-8);
        assert!((tick2.vy - 0.0).abs() < 5.0e-8);
        assert!(tick2.on_ground);
        assert!(tick2.alive);
    }

    #[test]
    fn living_support_block_uses_entity_movement_offset() {
        let mut region = region_with_shape([1, 3, 1]);
        region
            .set_block([0, 0, 0], &parse_block("minecraft:blue_ice"))
            .unwrap();
        region
            .set_block([0, 1, 0], &parse_block("minecraft:soul_sand"))
            .unwrap();

        let support =
            support_block_for(&region, Vec3d::new(0.5, 1.9, 0.5), VerifyEntityKind::Living)
                .expect("support block");
        assert_eq!(support.id, "soul_sand");
    }

    #[test]
    fn slab_runway_uses_block_below_for_item_friction() {
        let mut region = region_with_shape([4, 3, 1]);
        for x in 0..=3 {
            region
                .set_block([x, 0, 0], &parse_block("minecraft:blue_ice"))
                .unwrap();
            region
                .set_block(
                    [x, 1, 0],
                    &parse_block("minecraft:oak_slab[type=bottom,waterlogged=false]"),
                )
                .unwrap();
        }
        let world = LoadedSchematic {
            name: "slab-runway-blue-ice".to_string(),
            region,
            approximate_collision_blocks: Vec::new(),
        };
        let command = VerifyCommand {
            input: std::path::PathBuf::from("slab-runway-blue-ice.litematic"),
            out: std::path::PathBuf::from("artifacts/test"),
            target_speed: 0.0,
            ticks: 1,
            inspect_tick: Some(1),
            start_x: 0.5,
            start_y: 1.5,
            start_z: 0.5,
            start_vx: 1.0,
            start_vy: 0.0,
            start_vz: 0.0,
            start_on_ground: true,
            width: VERIFY_DEFAULT_WIDTH,
            height: VERIFY_DEFAULT_HEIGHT,
            entity_id_mod4: 0,
            initial_tick_count: 0,
            entity_rng_seed: None,
            entity_uuid: None,
            bootstrap_fluids: false,
            entity_kind: VerifyEntityKind::Item,
            no_ai: false,

            no_gravity: false,
            fire_immune: false,
            start_fire_ticks: 0,
            item_health: None,
        };

        let rows = simulate(&world, &command);
        let tick = &rows[1];
        let expected_vx = (0.989_f32 as f64) * HORIZONTAL_MOVEMENT_DAMPING;
        assert!((tick.x - 1.5).abs() < 1.0e-12);
        assert!((tick.y - 1.5).abs() < 1.0e-12);
        assert!((tick.vx - expected_vx).abs() < 1.0e-12);
        assert!(tick.on_ground);
        assert_eq!(tick.support_block, "minecraft:blue_ice");
    }

    #[test]
    fn slime_under_slab_does_not_apply_slime_step_on_drag() {
        let mut region = region_with_shape([4, 3, 1]);
        for x in 0..=3 {
            region
                .set_block([x, 0, 0], &parse_block("minecraft:slime_block"))
                .unwrap();
            region
                .set_block(
                    [x, 1, 0],
                    &parse_block("minecraft:oak_slab[type=bottom,waterlogged=false]"),
                )
                .unwrap();
        }
        let world = LoadedSchematic {
            name: "slab-runway-slime".to_string(),
            region,
            approximate_collision_blocks: Vec::new(),
        };
        let command = VerifyCommand {
            input: std::path::PathBuf::from("slab-runway-slime.litematic"),
            out: std::path::PathBuf::from("artifacts/test"),
            target_speed: 0.0,
            ticks: 1,
            inspect_tick: Some(1),
            start_x: 0.5,
            start_y: 1.5,
            start_z: 0.5,
            start_vx: 1.0,
            start_vy: 0.0,
            start_vz: 0.0,
            start_on_ground: true,
            width: VERIFY_DEFAULT_WIDTH,
            height: VERIFY_DEFAULT_HEIGHT,
            entity_id_mod4: 0,
            initial_tick_count: 0,
            entity_rng_seed: None,
            entity_uuid: None,
            bootstrap_fluids: false,
            entity_kind: VerifyEntityKind::Item,
            no_ai: false,

            no_gravity: false,
            fire_immune: false,
            start_fire_ticks: 0,
            item_health: None,
        };

        let rows = simulate(&world, &command);
        let tick = &rows[1];
        let expected_vx = (0.8_f32 as f64) * HORIZONTAL_MOVEMENT_DAMPING;
        assert!((tick.vx - expected_vx).abs() < 1.0e-12);
        assert_eq!(tick.support_block, "minecraft:slime_block");
    }

    #[test]
    fn slime_under_slab_does_not_bounce_item() {
        let mut region = region_with_shape([2, 4, 1]);
        for x in 0..=1 {
            region
                .set_block([x, 0, 0], &parse_block("minecraft:slime_block"))
                .unwrap();
            region
                .set_block(
                    [x, 1, 0],
                    &parse_block("minecraft:oak_slab[type=bottom,waterlogged=false]"),
                )
                .unwrap();
        }
        let world = LoadedSchematic {
            name: "slab-bounce-slime".to_string(),
            region,
            approximate_collision_blocks: Vec::new(),
        };
        let command = VerifyCommand {
            input: std::path::PathBuf::from("slab-bounce-slime.litematic"),
            out: std::path::PathBuf::from("artifacts/test"),
            target_speed: 0.0,
            ticks: 1,
            inspect_tick: Some(1),
            start_x: 0.5,
            start_y: 2.0,
            start_z: 0.5,
            start_vx: 0.0,
            start_vy: -1.0,
            start_vz: 0.0,
            start_on_ground: false,
            width: VERIFY_DEFAULT_WIDTH,
            height: VERIFY_DEFAULT_HEIGHT,
            entity_id_mod4: 0,
            initial_tick_count: 0,
            entity_rng_seed: None,
            entity_uuid: None,
            bootstrap_fluids: false,
            entity_kind: VerifyEntityKind::Item,
            no_ai: false,

            no_gravity: false,
            fire_immune: false,
            start_fire_ticks: 0,
            item_health: None,
        };

        let rows = simulate(&world, &command);
        let tick = &rows[1];
        assert!((tick.y - 1.5).abs() < 1.0e-12);
        assert!((tick.vy - 0.0).abs() < 1.0e-12);
        assert!(tick.on_ground);
    }

    #[test]
    fn simulate_item_slime_bounce_matches_vanilla_probe() {
        let mut region = region_with_shape([3, 3, 3]);
        region
            .set_block([1, 0, 1], &parse_block("minecraft:slime_block"))
            .unwrap();
        let world = LoadedSchematic {
            name: "slime-bounce-vanilla".to_string(),
            region,
            approximate_collision_blocks: Vec::new(),
        };
        let command = VerifyCommand {
            input: std::path::PathBuf::from("slime-bounce-vanilla.litematic"),
            out: std::path::PathBuf::from("artifacts/test"),
            target_speed: 0.0,
            ticks: 1,
            inspect_tick: Some(1),
            start_x: 1.5,
            start_y: 1.2,
            start_z: 1.5,
            start_vx: 0.0,
            start_vy: -1.0,
            start_vz: 0.0,
            start_on_ground: false,
            width: VERIFY_DEFAULT_WIDTH,
            height: VERIFY_DEFAULT_HEIGHT,
            entity_id_mod4: 0,
            initial_tick_count: 0,
            entity_rng_seed: None,
            entity_uuid: None,
            bootstrap_fluids: false,
            entity_kind: VerifyEntityKind::Item,
            no_ai: false,
            no_gravity: false,
            fire_immune: false,
            start_fire_ticks: 0,
            item_health: None,
        };

        let rows = simulate(&world, &command);
        let tick = &rows[1];
        assert!((tick.x - 1.5).abs() < 5.0e-8);
        assert!((tick.y - 1.0).abs() < 5.0e-8);
        assert!((tick.z - 1.5).abs() < 5.0e-8);
        assert!((tick.vx - 0.0).abs() < 5.0e-8);
        assert!((tick.vy - 0.818_231_605_094_970_7).abs() < 5.0e-8);
        assert!((tick.vz - 0.0).abs() < 5.0e-8);
        assert!(tick.on_ground);
        assert!(tick.alive);
        assert!(tick.removed_by.is_empty());
    }

    #[test]
    fn soul_sand_mud_and_honey_collision_boxes_match_vanilla_shapes() {
        let soul_sand = parse_block("minecraft:soul_sand");
        let mud = parse_block("minecraft:mud");
        let honey = parse_block("minecraft:honey_block");

        let soul_shape = collision_boxes(&soul_sand)
            .iter()
            .next()
            .copied()
            .expect("soul sand shape");
        assert!((soul_shape.max_y - 14.0 / 16.0).abs() < 1.0e-12);
        assert!((soul_shape.min_x - 0.0).abs() < 1.0e-12);
        assert!((soul_shape.max_x - 1.0).abs() < 1.0e-12);

        let mud_shape = collision_boxes(&mud)
            .iter()
            .next()
            .copied()
            .expect("mud shape");
        assert!((mud_shape.max_y - 14.0 / 16.0).abs() < 1.0e-12);
        assert!((mud_shape.min_x - 0.0).abs() < 1.0e-12);
        assert!((mud_shape.max_x - 1.0).abs() < 1.0e-12);

        let honey_shape = collision_boxes(&honey)
            .iter()
            .next()
            .copied()
            .expect("honey shape");
        assert!((honey_shape.min_x - 1.0 / 16.0).abs() < 1.0e-12);
        assert!((honey_shape.max_x - 15.0 / 16.0).abs() < 1.0e-12);
        assert!((honey_shape.max_y - 15.0 / 16.0).abs() < 1.0e-12);
    }

    #[test]
    fn cauldron_family_collision_boxes_match_guardian_shapes() {
        for block_id in [
            "minecraft:cauldron",
            "minecraft:water_cauldron[level=1]",
            "minecraft:lava_cauldron",
            "minecraft:powder_snow_cauldron[level=3]",
        ] {
            let block = parse_block(block_id);
            assert_eq!(collision_kind(&block), CollisionKind::PartialBlock);
            let boxes: Vec<_> = collision_boxes(&block).iter().copied().collect();
            assert_eq!(
                boxes.len(),
                8,
                "unexpected cauldron box count for {block_id}"
            );
            assert!(boxes.iter().any(|shape| {
                (shape.min_x - 0.0).abs() < 1.0e-12
                    && (shape.max_x - 2.0 / 16.0).abs() < 1.0e-12
                    && (shape.min_z - 0.0).abs() < 1.0e-12
                    && (shape.max_z - 2.0 / 16.0).abs() < 1.0e-12
                    && (shape.max_y - 3.0 / 16.0).abs() < 1.0e-12
            }));
            assert!(boxes.iter().any(|shape| {
                (shape.min_z - 0.0).abs() < 1.0e-12
                    && (shape.max_z - 2.0 / 16.0).abs() < 1.0e-12
                    && (shape.min_y - 3.0 / 16.0).abs() < 1.0e-12
                    && (shape.max_y - 1.0).abs() < 1.0e-12
                    && (shape.min_x - 0.0).abs() < 1.0e-12
                    && (shape.max_x - 1.0).abs() < 1.0e-12
            }));
        }
    }

    #[test]
    fn composter_lectern_bell_and_anvil_collision_boxes_match_guardian_shapes() {
        let composter = parse_block("minecraft:composter[level=4]");
        assert_eq!(collision_kind(&composter), CollisionKind::PartialBlock);
        let composter_boxes: Vec<_> = collision_boxes(&composter).iter().copied().collect();
        assert_eq!(composter_boxes.len(), 5);
        assert!((composter_boxes[0].max_y - 2.0 / 16.0).abs() < 1.0e-12);
        assert!((composter_boxes[0].max_x - 1.0).abs() < 1.0e-12);

        let lectern = parse_block("minecraft:lectern[facing=north,has_book=false,powered=false]");
        assert_eq!(collision_kind(&lectern), CollisionKind::PartialBlock);
        let lectern_boxes: Vec<_> = collision_boxes(&lectern).iter().copied().collect();
        assert_eq!(lectern_boxes.len(), 2);
        assert!((lectern_boxes[0].max_y - 2.0 / 16.0).abs() < 1.0e-12);
        assert!((lectern_boxes[1].min_x - 4.0 / 16.0).abs() < 1.0e-12);
        assert!((lectern_boxes[1].max_x - 12.0 / 16.0).abs() < 1.0e-12);
        assert!((lectern_boxes[1].max_y - 14.0 / 16.0).abs() < 1.0e-12);

        let bell = parse_block("minecraft:bell[attachment=single_wall,facing=north,powered=false]");
        assert_eq!(collision_kind(&bell), CollisionKind::PartialBlock);
        let bell_boxes: Vec<_> = collision_boxes(&bell).iter().copied().collect();
        assert_eq!(bell_boxes.len(), 3);
        assert!(bell_boxes.iter().any(|shape| {
            (shape.min_x - 7.0 / 16.0).abs() < 1.0e-12
                && (shape.max_x - 9.0 / 16.0).abs() < 1.0e-12
                && (shape.min_z - 0.0).abs() < 1.0e-12
                && (shape.max_z - 13.0 / 16.0).abs() < 1.0e-12
                && (shape.max_y - 15.0 / 16.0).abs() < 1.0e-12
        }));

        let anvil_north = parse_block("minecraft:anvil[facing=north]");
        assert_eq!(collision_kind(&anvil_north), CollisionKind::PartialBlock);
        let anvil_north_boxes: Vec<_> = collision_boxes(&anvil_north).iter().copied().collect();
        assert_eq!(anvil_north_boxes.len(), 4);
        assert!(anvil_north_boxes.iter().any(|shape| {
            (shape.min_x - 3.0 / 16.0).abs() < 1.0e-12
                && (shape.max_x - 13.0 / 16.0).abs() < 1.0e-12
                && (shape.min_z - 0.0).abs() < 1.0e-12
                && (shape.max_z - 1.0).abs() < 1.0e-12
                && (shape.min_y - 10.0 / 16.0).abs() < 1.0e-12
                && (shape.max_y - 1.0).abs() < 1.0e-12
        }));

        let anvil_east = parse_block("minecraft:anvil[facing=east]");
        let anvil_east_boxes: Vec<_> = collision_boxes(&anvil_east).iter().copied().collect();
        assert!(anvil_east_boxes.iter().any(|shape| {
            (shape.min_x - 0.0).abs() < 1.0e-12
                && (shape.max_x - 1.0).abs() < 1.0e-12
                && (shape.min_z - 3.0 / 16.0).abs() < 1.0e-12
                && (shape.max_z - 13.0 / 16.0).abs() < 1.0e-12
                && (shape.min_y - 10.0 / 16.0).abs() < 1.0e-12
                && (shape.max_y - 1.0).abs() < 1.0e-12
        }));
    }

    #[test]
    fn candle_and_amethyst_collision_classification_matches_guardian() {
        let candle = parse_block("minecraft:candle[candles=4,lit=false,waterlogged=false]");
        assert_eq!(collision_kind(&candle), CollisionKind::PartialBlock);
        let candle_shape = collision_boxes(&candle)
            .iter()
            .next()
            .copied()
            .expect("candle shape");
        assert!((candle_shape.min_x - 5.0 / 16.0).abs() < 1.0e-12);
        assert!((candle_shape.max_x - 11.0 / 16.0).abs() < 1.0e-12);
        assert!((candle_shape.max_y - 6.0 / 16.0).abs() < 1.0e-12);

        let amethyst_block = parse_block("minecraft:amethyst_block");
        assert_eq!(collision_kind(&amethyst_block), CollisionKind::FullBlock);

        let budding_amethyst = parse_block("minecraft:budding_amethyst");
        assert_eq!(collision_kind(&budding_amethyst), CollisionKind::FullBlock);

        let cluster = parse_block("minecraft:amethyst_cluster[facing=up,waterlogged=false]");
        assert_eq!(collision_kind(&cluster), CollisionKind::PartialBlock);
        let cluster_shape = collision_boxes(&cluster)
            .iter()
            .next()
            .copied()
            .expect("cluster shape");
        assert!((cluster_shape.min_y - 0.0).abs() < 1.0e-12);
        assert!((cluster_shape.max_y - 7.0 / 16.0).abs() < 1.0e-12);
        assert!((cluster_shape.min_x - 3.0 / 16.0).abs() < 1.0e-12);
        assert!((cluster_shape.max_x - 13.0 / 16.0).abs() < 1.0e-12);
    }

    #[test]
    fn big_dripleaf_collision_boxes_follow_tilt_state() {
        let stable =
            parse_block("minecraft:big_dripleaf[facing=north,tilt=none,waterlogged=false]");
        assert_eq!(collision_kind(&stable), CollisionKind::PartialBlock);
        let stable_shape = collision_boxes(&stable)
            .iter()
            .next()
            .copied()
            .expect("stable big dripleaf shape");
        assert!((stable_shape.min_y - 11.0 / 16.0).abs() < 1.0e-12);
        assert!((stable_shape.max_y - 15.0 / 16.0).abs() < 1.0e-12);
        assert!((stable_shape.min_x - 0.0).abs() < 1.0e-12);
        assert!((stable_shape.max_x - 1.0).abs() < 1.0e-12);

        let partial =
            parse_block("minecraft:big_dripleaf[facing=north,tilt=partial,waterlogged=false]");
        let partial_shape = collision_boxes(&partial)
            .iter()
            .next()
            .copied()
            .expect("partial big dripleaf shape");
        assert!((partial_shape.min_y - 11.0 / 16.0).abs() < 1.0e-12);
        assert!((partial_shape.max_y - 13.0 / 16.0).abs() < 1.0e-12);

        let full = parse_block("minecraft:big_dripleaf[facing=north,tilt=full,waterlogged=false]");
        assert_eq!(collision_kind(&full), CollisionKind::PartialBlock);
        assert!(collision_boxes(&full).is_empty());
    }

    #[test]
    fn pointed_dripstone_collision_boxes_follow_thickness_and_direction() {
        let tip_up = parse_block(
            "minecraft:pointed_dripstone[thickness=tip,vertical_direction=up,waterlogged=false]",
        );
        assert_eq!(collision_kind(&tip_up), CollisionKind::PartialBlock);
        let tip_up_shape = collision_boxes(&tip_up)
            .iter()
            .next()
            .copied()
            .expect("tip-up dripstone shape");
        assert!((tip_up_shape.min_x - 5.0 / 16.0).abs() < 1.0e-12);
        assert!((tip_up_shape.max_x - 11.0 / 16.0).abs() < 1.0e-12);
        assert!((tip_up_shape.min_y - 0.0).abs() < 1.0e-12);
        assert!((tip_up_shape.max_y - 11.0 / 16.0).abs() < 1.0e-12);

        let tip_down = parse_block(
            "minecraft:pointed_dripstone[thickness=tip,vertical_direction=down,waterlogged=false]",
        );
        let tip_down_shape = collision_boxes(&tip_down)
            .iter()
            .next()
            .copied()
            .expect("tip-down dripstone shape");
        assert!((tip_down_shape.min_y - 5.0 / 16.0).abs() < 1.0e-12);
        assert!((tip_down_shape.max_y - 1.0).abs() < 1.0e-12);

        let middle = parse_block(
            "minecraft:pointed_dripstone[thickness=middle,vertical_direction=up,waterlogged=false]",
        );
        let middle_shape = collision_boxes(&middle)
            .iter()
            .next()
            .copied()
            .expect("middle dripstone shape");
        assert!((middle_shape.min_x - 3.0 / 16.0).abs() < 1.0e-12);
        assert!((middle_shape.max_x - 13.0 / 16.0).abs() < 1.0e-12);
        assert!((middle_shape.max_y - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn lever_and_coral_variants_are_non_solid_but_coral_blocks_are_full() {
        let lever = parse_block("minecraft:lever[face=wall,facing=north,powered=false]");
        assert_eq!(collision_kind(&lever), CollisionKind::NonSolid);
        assert!(collision_boxes(&lever).is_empty());

        let tube_coral = parse_block("minecraft:tube_coral[waterlogged=true]");
        assert_eq!(collision_kind(&tube_coral), CollisionKind::NonSolid);
        assert!(collision_boxes(&tube_coral).is_empty());

        let dead_tube_coral_fan = parse_block("minecraft:dead_tube_coral_fan[waterlogged=false]");
        assert_eq!(
            collision_kind(&dead_tube_coral_fan),
            CollisionKind::NonSolid
        );
        assert!(collision_boxes(&dead_tube_coral_fan).is_empty());

        let brain_coral_block = parse_block("minecraft:brain_coral_block");
        assert_eq!(collision_kind(&brain_coral_block), CollisionKind::FullBlock);
        let full_shape = collision_boxes(&brain_coral_block)
            .iter()
            .next()
            .copied()
            .expect("full coral block shape");
        assert!((full_shape.max_x - 1.0).abs() < 1.0e-12);
        assert!((full_shape.max_y - 1.0).abs() < 1.0e-12);
        assert!((full_shape.max_z - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn guardian_no_collision_plant_families_stay_non_solid() {
        for block_id in [
            "minecraft:oak_sapling[stage=0]",
            "minecraft:mangrove_propagule[age=0,hanging=false,stage=0,waterlogged=false]",
            "minecraft:poppy",
            "minecraft:blue_orchid",
            "minecraft:red_tulip",
            "minecraft:wheat[age=7]",
            "minecraft:beetroots[age=3]",
            "minecraft:nether_wart[age=3]",
            "minecraft:pumpkin_stem[age=7]",
            "minecraft:attached_melon_stem[facing=north]",
            "minecraft:cave_vines[berries=true,age=25]",
            "minecraft:spore_blossom",
            "minecraft:pink_petals[flower_amount=4,facing=north]",
            "minecraft:wildflowers[flower_amount=4,facing=north]",
            "minecraft:leaf_litter[segment_amount=4,facing=north,waterlogged=false]",
            "minecraft:small_dripleaf[facing=north,half=lower,waterlogged=false]",
            "minecraft:big_dripleaf_stem[facing=north,waterlogged=false]",
            "minecraft:hanging_roots[waterlogged=false]",
            "minecraft:warped_roots",
            "minecraft:nether_sprouts",
        ] {
            let block = parse_block(block_id);
            assert_eq!(
                collision_kind(&block),
                CollisionKind::NonSolid,
                "expected non-solid collision for {block_id}"
            );
            assert!(
                collision_boxes(&block).is_empty(),
                "expected empty collision boxes for {block_id}"
            );
        }
    }

    #[test]
    fn simple_environment_partials_match_guardian_shapes() {
        let sea_pickle = parse_block("minecraft:sea_pickle[pickles=4,waterlogged=true]");
        assert_eq!(collision_kind(&sea_pickle), CollisionKind::PartialBlock);
        let sea_pickle_shape = collision_boxes(&sea_pickle)
            .iter()
            .next()
            .copied()
            .expect("sea pickle shape");
        assert!((sea_pickle_shape.min_x - 2.0 / 16.0).abs() < 1.0e-12);
        assert!((sea_pickle_shape.max_x - 14.0 / 16.0).abs() < 1.0e-12);
        assert!((sea_pickle_shape.max_y - 7.0 / 16.0).abs() < 1.0e-12);

        let lily_pad = parse_block("minecraft:lily_pad");
        assert_eq!(collision_kind(&lily_pad), CollisionKind::PartialBlock);
        let lily_pad_shape = collision_boxes(&lily_pad)
            .iter()
            .next()
            .copied()
            .expect("lily pad shape");
        assert!((lily_pad_shape.min_x - 1.0 / 16.0).abs() < 1.0e-12);
        assert!((lily_pad_shape.max_x - 15.0 / 16.0).abs() < 1.0e-12);
        assert!((lily_pad_shape.max_y - 1.5 / 16.0).abs() < 1.0e-12);

        let frogspawn = parse_block("minecraft:frogspawn");
        assert_eq!(collision_kind(&frogspawn), CollisionKind::PartialBlock);
        let frogspawn_shape = collision_boxes(&frogspawn)
            .iter()
            .next()
            .copied()
            .expect("frogspawn shape");
        assert!((frogspawn_shape.max_x - 1.0).abs() < 1.0e-12);
        assert!((frogspawn_shape.max_y - 1.5 / 16.0).abs() < 1.0e-12);

        let turtle_egg = parse_block("minecraft:turtle_egg[eggs=1,hatch=0]");
        assert_eq!(collision_kind(&turtle_egg), CollisionKind::PartialBlock);
        let turtle_egg_shape = collision_boxes(&turtle_egg)
            .iter()
            .next()
            .copied()
            .expect("turtle egg shape");
        assert!((turtle_egg_shape.min_x - 3.0 / 16.0).abs() < 1.0e-12);
        assert!((turtle_egg_shape.max_x - 12.0 / 16.0).abs() < 1.0e-12);
        assert!((turtle_egg_shape.max_y - 7.0 / 16.0).abs() < 1.0e-12);

        let potted_poppy = parse_block("minecraft:potted_poppy");
        assert_eq!(collision_kind(&potted_poppy), CollisionKind::PartialBlock);
        let potted_poppy_shape = collision_boxes(&potted_poppy)
            .iter()
            .next()
            .copied()
            .expect("potted poppy shape");
        assert!((potted_poppy_shape.min_x - 5.0 / 16.0).abs() < 1.0e-12);
        assert!((potted_poppy_shape.max_x - 11.0 / 16.0).abs() < 1.0e-12);
        assert!((potted_poppy_shape.max_y - 6.0 / 16.0).abs() < 1.0e-12);
    }

    #[test]
    fn functional_partial_blocks_match_guardian_shapes() {
        let brewing_stand = parse_block(
            "minecraft:brewing_stand[has_bottle_0=false,has_bottle_1=false,has_bottle_2=false]",
        );
        assert_eq!(collision_kind(&brewing_stand), CollisionKind::PartialBlock);
        let brewing_boxes: Vec<_> = collision_boxes(&brewing_stand).iter().copied().collect();
        assert_eq!(brewing_boxes.len(), 2);
        assert!(brewing_boxes.iter().any(|shape| {
            (shape.min_x - 7.0 / 16.0).abs() < 1.0e-12
                && (shape.max_x - 9.0 / 16.0).abs() < 1.0e-12
                && (shape.min_y - 2.0 / 16.0).abs() < 1.0e-12
                && (shape.max_y - 14.0 / 16.0).abs() < 1.0e-12
        }));

        let conduit = parse_block("minecraft:conduit[waterlogged=true]");
        assert_eq!(collision_kind(&conduit), CollisionKind::PartialBlock);
        let conduit_shape = collision_boxes(&conduit)
            .iter()
            .next()
            .copied()
            .expect("conduit shape");
        assert!((conduit_shape.min_x - 5.0 / 16.0).abs() < 1.0e-12);
        assert!((conduit_shape.max_x - 11.0 / 16.0).abs() < 1.0e-12);
        assert!((conduit_shape.min_y - 5.0 / 16.0).abs() < 1.0e-12);
        assert!((conduit_shape.max_y - 11.0 / 16.0).abs() < 1.0e-12);

        let end_portal_frame = parse_block("minecraft:end_portal_frame[facing=north,eye=true]");
        assert_eq!(
            collision_kind(&end_portal_frame),
            CollisionKind::PartialBlock
        );
        let end_portal_boxes: Vec<_> = collision_boxes(&end_portal_frame).iter().copied().collect();
        assert_eq!(end_portal_boxes.len(), 2);
        assert!(end_portal_boxes.iter().any(|shape| {
            (shape.max_y - 13.0 / 16.0).abs() < 1.0e-12
                && (shape.max_x - 1.0).abs() < 1.0e-12
                && (shape.max_z - 1.0).abs() < 1.0e-12
        }));
        assert!(end_portal_boxes.iter().any(|shape| {
            (shape.min_x - 4.0 / 16.0).abs() < 1.0e-12
                && (shape.max_x - 12.0 / 16.0).abs() < 1.0e-12
                && (shape.min_y - 13.0 / 16.0).abs() < 1.0e-12
                && (shape.max_y - 1.0).abs() < 1.0e-12
        }));

        let cocoa = parse_block("minecraft:cocoa[age=2,facing=north]");
        assert_eq!(collision_kind(&cocoa), CollisionKind::PartialBlock);
        let cocoa_shape = collision_boxes(&cocoa)
            .iter()
            .next()
            .copied()
            .expect("cocoa shape");
        assert!((cocoa_shape.min_x - 4.0 / 16.0).abs() < 1.0e-12);
        assert!((cocoa_shape.max_x - 12.0 / 16.0).abs() < 1.0e-12);
        assert!((cocoa_shape.min_y - 3.0 / 16.0).abs() < 1.0e-12);
        assert!((cocoa_shape.max_y - 12.0 / 16.0).abs() < 1.0e-12);
        assert!((cocoa_shape.min_z - 1.0 / 16.0).abs() < 1.0e-12);
        assert!((cocoa_shape.max_z - 9.0 / 16.0).abs() < 1.0e-12);
    }

    #[test]
    fn table_and_cutting_blocks_match_guardian_shapes() {
        let stonecutter = parse_block("minecraft:stonecutter[facing=north]");
        assert_eq!(collision_kind(&stonecutter), CollisionKind::PartialBlock);
        let stonecutter_shape = collision_boxes(&stonecutter)
            .iter()
            .next()
            .copied()
            .expect("stonecutter shape");
        assert!((stonecutter_shape.max_y - 9.0 / 16.0).abs() < 1.0e-12);
        assert!((stonecutter_shape.max_x - 1.0).abs() < 1.0e-12);

        let cake = parse_block("minecraft:cake[bites=3]");
        assert_eq!(collision_kind(&cake), CollisionKind::PartialBlock);
        let cake_shape = collision_boxes(&cake)
            .iter()
            .next()
            .copied()
            .expect("cake shape");
        assert!((cake_shape.min_x - 7.0 / 16.0).abs() < 1.0e-12);
        assert!((cake_shape.max_y - 8.0 / 16.0).abs() < 1.0e-12);
        assert!((cake_shape.max_z - 15.0 / 16.0).abs() < 1.0e-12);

        let enchanting_table = parse_block("minecraft:enchanting_table");
        assert_eq!(
            collision_kind(&enchanting_table),
            CollisionKind::PartialBlock
        );
        let enchanting_shape = collision_boxes(&enchanting_table)
            .iter()
            .next()
            .copied()
            .expect("enchanting table shape");
        assert!((enchanting_shape.max_y - 12.0 / 16.0).abs() < 1.0e-12);
        assert!((enchanting_shape.max_x - 1.0).abs() < 1.0e-12);

        let daylight_detector = parse_block("minecraft:daylight_detector[inverted=false,power=0]");
        assert_eq!(
            collision_kind(&daylight_detector),
            CollisionKind::PartialBlock
        );
        let daylight_shape = collision_boxes(&daylight_detector)
            .iter()
            .next()
            .copied()
            .expect("daylight detector shape");
        assert!((daylight_shape.max_y - 6.0 / 16.0).abs() < 1.0e-12);
        assert!((daylight_shape.max_x - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn waterlogged_blocks_keep_collision_shapes_instead_of_becoming_fluids() {
        for block_id in [
            "minecraft:sea_pickle[pickles=1,waterlogged=true]",
            "minecraft:scaffolding[bottom=false,distance=0,waterlogged=true]",
            "minecraft:oak_trapdoor[facing=north,half=bottom,open=false,powered=false,waterlogged=true]",
            "minecraft:ladder[facing=north,waterlogged=true]",
        ] {
            let block = parse_block(block_id);
            assert_ne!(
                collision_kind(&block),
                CollisionKind::NonSolid,
                "waterlogged block lost its collision classification for {block_id}"
            );
            assert!(
                !collision_boxes(&block).is_empty(),
                "waterlogged block lost its collision boxes for {block_id}"
            );
        }
    }

    #[test]
    fn bamboo_blocks_match_guardian_collision_and_support_rules() {
        let bamboo = parse_block("minecraft:bamboo[age=0,leaves=none,stage=0]");
        assert_eq!(collision_kind(&bamboo), CollisionKind::PartialBlock);
        let bamboo_shape = collision_boxes(&bamboo)
            .iter()
            .next()
            .copied()
            .expect("bamboo collision shape");
        assert!((bamboo_shape.min_x - 13.0 / 32.0).abs() < 1.0e-12);
        assert!((bamboo_shape.max_x - 19.0 / 32.0).abs() < 1.0e-12);
        assert!((bamboo_shape.max_y - 1.0).abs() < 1.0e-12);

        let seed = guardian_block_seed(1, 0, 1) as u64;
        let expected_x = (((((seed & 15) as f32) / 15.0_f32) - 0.5_f32) * 0.5_f32)
            .clamp(-0.25_f32, 0.25_f32) as f64;
        let expected_z = ((((((seed >> 8) & 15) as f32) / 15.0_f32) - 0.5_f32) * 0.5_f32)
            .clamp(-0.25_f32, 0.25_f32) as f64;
        let shifted = world_collision_box(&bamboo, [1, 0, 1], &bamboo_shape);
        assert!((shifted.min_x - (1.0 + bamboo_shape.min_x + expected_x)).abs() < 1.0e-12);
        assert!((shifted.min_z - (1.0 + bamboo_shape.min_z + expected_z)).abs() < 1.0e-12);

        let sapling = parse_block("minecraft:bamboo_sapling");
        assert_eq!(collision_kind(&sapling), CollisionKind::NonSolid);
        assert!(collision_boxes(&sapling).is_empty());

        let mut supported = region_with_shape([1, 2, 1]);
        supported
            .set_block([0, 0, 0], &parse_block("minecraft:sand"))
            .unwrap();
        supported.set_block([0, 1, 0], &bamboo).unwrap();
        run_world_ticks(&mut supported, 1);
        assert_eq!(
            block_full_id(block_at(&supported, [0, 1, 0]).expect("supported bamboo remains")),
            "minecraft:bamboo[age=0,leaves=none,stage=0]"
        );

        let mut unsupported = region_with_shape([1, 2, 1]);
        unsupported.set_block([0, 1, 0], &bamboo).unwrap();
        run_world_ticks(&mut unsupported, 2);
        assert_eq!(
            block_full_id(
                block_at(&unsupported, [0, 1, 0])
                    .expect("unsupported bamboo persists through two snapshot ticks")
            ),
            "minecraft:bamboo[age=0,leaves=none,stage=0]"
        );

        let mut scheduled_break = region_with_shape([1, 2, 1]);
        scheduled_break.set_block([0, 1, 0], &bamboo).unwrap();
        let mut block_ticks = DynamicBlockTicks::default();
        block_ticks.schedule(BAMBOO_TICK_DELAY, [0, 1, 0]);
        block_ticks.run_due(&mut scheduled_break, BAMBOO_TICK_DELAY);
        assert_eq!(
            block_full_id(
                block_at(&scheduled_break, [0, 1, 0]).expect("scheduled bamboo break yields air")
            ),
            "minecraft:air"
        );
    }

    #[test]
    fn loaded_snapshot_prunes_reload_invalid_blocks_but_keeps_reload_stable_blocks() {
        let mut bamboo_region = region_with_shape([1, 2, 1]);
        bamboo_region
            .set_block(
                [0, 1, 0],
                &parse_block("minecraft:bamboo[age=0,leaves=none,stage=0]"),
            )
            .unwrap();
        normalize_loaded_snapshot(&mut bamboo_region);
        assert_eq!(
            block_full_id(
                block_at(&bamboo_region, [0, 1, 0]).expect("unsupported bamboo pruned on load")
            ),
            "minecraft:air"
        );

        let mut cactus_region = region_with_shape([1, 2, 1]);
        cactus_region
            .set_block([0, 1, 0], &parse_block("minecraft:cactus[age=0]"))
            .unwrap();
        normalize_loaded_snapshot(&mut cactus_region);
        assert_eq!(
            block_full_id(
                block_at(&cactus_region, [0, 1, 0]).expect("unsupported cactus pruned on load")
            ),
            "minecraft:air"
        );

        let mut scaffolding_region = region_with_shape([1, 2, 1]);
        scaffolding_region
            .set_block(
                [0, 1, 0],
                &parse_block("minecraft:scaffolding[bottom=false,distance=0,waterlogged=false]"),
            )
            .unwrap();
        normalize_loaded_snapshot(&mut scaffolding_region);
        assert_eq!(
            block_full_id(
                block_at(&scaffolding_region, [0, 1, 0])
                    .expect("unsupported scaffolding pruned on load")
            ),
            "minecraft:air"
        );

        let mut dripstone_region = region_with_shape([1, 2, 1]);
        dripstone_region
            .set_block(
                [0, 1, 0],
                &parse_block(
                    "minecraft:pointed_dripstone[thickness=tip,vertical_direction=up,waterlogged=false]",
                ),
            )
            .unwrap();
        normalize_loaded_snapshot(&mut dripstone_region);
        assert_eq!(
            block_full_id(
                block_at(&dripstone_region, [0, 1, 0])
                    .expect("unsupported pointed dripstone pruned on load")
            ),
            "minecraft:air"
        );

        let mut dripleaf_region = region_with_shape([1, 2, 1]);
        dripleaf_region
            .set_block(
                [0, 1, 0],
                &parse_block("minecraft:big_dripleaf[facing=north,tilt=none,waterlogged=false]"),
            )
            .unwrap();
        normalize_loaded_snapshot(&mut dripleaf_region);
        assert_eq!(
            block_full_id(
                block_at(&dripleaf_region, [0, 1, 0])
                    .expect("unsupported big dripleaf still loads from snapshot")
            ),
            "minecraft:big_dripleaf[facing=north,tilt=none,waterlogged=false]"
        );

        let mut dripleaf_stem_region = region_with_shape([1, 3, 1]);
        dripleaf_stem_region
            .set_block(
                [0, 1, 0],
                &parse_block("minecraft:big_dripleaf_stem[facing=north,waterlogged=false]"),
            )
            .unwrap();
        normalize_loaded_snapshot(&mut dripleaf_stem_region);
        assert_eq!(
            block_full_id(
                block_at(&dripleaf_stem_region, [0, 1, 0])
                    .expect("unsupported big dripleaf stem pruned on load")
            ),
            "minecraft:air"
        );

        let mut dripleaf_chain_region = region_with_shape([1, 3, 1]);
        dripleaf_chain_region
            .set_block(
                [0, 1, 0],
                &parse_block("minecraft:big_dripleaf_stem[facing=north,waterlogged=false]"),
            )
            .unwrap();
        dripleaf_chain_region
            .set_block(
                [0, 2, 0],
                &parse_block("minecraft:big_dripleaf[facing=north,tilt=none,waterlogged=false]"),
            )
            .unwrap();
        normalize_loaded_snapshot(&mut dripleaf_chain_region);
        assert_eq!(
            block_full_id(
                block_at(&dripleaf_chain_region, [0, 1, 0])
                    .expect("unsupported big dripleaf stem pruned from chain on load")
            ),
            "minecraft:air"
        );
        assert_eq!(
            block_full_id(
                block_at(&dripleaf_chain_region, [0, 2, 0])
                    .expect("leaf above unsupported stem pruned on load")
            ),
            "minecraft:air"
        );

        let mut ladder_region = region_with_shape([1, 2, 1]);
        ladder_region
            .set_block(
                [0, 1, 0],
                &parse_block("minecraft:ladder[facing=north,waterlogged=false]"),
            )
            .unwrap();
        normalize_loaded_snapshot(&mut ladder_region);
        assert_eq!(
            block_full_id(
                block_at(&ladder_region, [0, 1, 0])
                    .expect("unsupported ladder still loads from snapshot")
            ),
            "minecraft:ladder[facing=north,waterlogged=false]"
        );
    }

    #[test]
    fn item_lands_on_stonecutter_and_cake_heights() {
        let cases = [
            (
                "minecraft:stonecutter[facing=north]",
                9.0 / 16.0,
                "stonecutter-landing",
            ),
            ("minecraft:cake[bites=0]", 8.0 / 16.0, "cake-landing"),
        ];

        for (block_id, expected_y, name) in cases {
            let mut region = region_with_shape([3, 3, 3]);
            region.set_block([1, 0, 1], &parse_block(block_id)).unwrap();
            let world = LoadedSchematic {
                name: name.to_string(),
                region,
                approximate_collision_blocks: Vec::new(),
            };
            let command = VerifyCommand {
                input: std::path::PathBuf::from(format!("{name}.litematic")),
                out: std::path::PathBuf::from("artifacts/test"),
                target_speed: 0.0,
                ticks: 3,
                inspect_tick: Some(3),
                start_x: 1.5,
                start_y: 1.3,
                start_z: 1.5,
                start_vx: 0.0,
                start_vy: -0.2,
                start_vz: 0.0,
                start_on_ground: false,
                width: VERIFY_DEFAULT_WIDTH,
                height: VERIFY_DEFAULT_HEIGHT,
                entity_id_mod4: 0,
                initial_tick_count: 0,
                entity_rng_seed: None,
                entity_uuid: None,
                bootstrap_fluids: false,
                entity_kind: VerifyEntityKind::Item,
                no_ai: false,
                no_gravity: false,
                fire_immune: false,
                start_fire_ticks: 0,
                item_health: None,
            };

            let rows = simulate(&world, &command);
            let tick3 = &rows[3];
            assert!(tick3.on_ground, "expected landing on {block_id}");
            assert!(
                (tick3.y - expected_y).abs() < 5.0e-8,
                "unexpected landing height for {block_id}"
            );
            assert!(
                (tick3.vy - 0.0).abs() < 5.0e-8,
                "expected zero vertical speed on {block_id}"
            );
            assert!(tick3.alive);
        }
    }

    #[test]
    fn rails_are_non_solid_for_entity_collision() {
        let flat_rail = parse_block("minecraft:rail[shape=north_south,waterlogged=false]");
        assert_eq!(collision_kind(&flat_rail), CollisionKind::NonSolid);
        assert!(collision_boxes(&flat_rail).is_empty());

        let slope_rail = parse_block("minecraft:rail[shape=ascending_east,waterlogged=false]");
        assert_eq!(collision_kind(&slope_rail), CollisionKind::NonSolid);
        assert!(collision_boxes(&slope_rail).is_empty());
    }

    #[test]
    fn soul_sand_and_mud_support_scaffolding_like_vanilla() {
        for floor_block in [
            "minecraft:soul_sand",
            "minecraft:mud",
            "minecraft:oak_slab[type=bottom,waterlogged=false]",
        ] {
            let mut region = region_with_shape([1, 3, 1]);
            region
                .set_block([0, 0, 0], &parse_block(floor_block))
                .unwrap();
            region
                .set_block(
                    [0, 1, 0],
                    &parse_block(
                        "minecraft:scaffolding[bottom=false,distance=0,waterlogged=false]",
                    ),
                )
                .unwrap();
            run_world_ticks(&mut region, 1);
            assert_eq!(
                block_full_id(block_at(&region, [0, 1, 0]).expect("supported scaffolding remains")),
                "minecraft:scaffolding[bottom=false,distance=0,waterlogged=false]",
                "unexpected scaffolding support result for {floor_block}"
            );
        }
    }

    #[test]
    fn snow_layer_support_shape_matches_vanilla_full_face_rules() {
        for snow_block in ["minecraft:snow[layers=8]", "minecraft:snow[layers=7]"] {
            let mut region = region_with_shape([1, 3, 1]);
            region
                .set_block([0, 0, 0], &parse_block(snow_block))
                .unwrap();
            region
                .set_block(
                    [0, 1, 0],
                    &parse_block(
                        "minecraft:scaffolding[bottom=false,distance=0,waterlogged=false]",
                    ),
                )
                .unwrap();
            run_world_ticks(&mut region, 1);
            assert_eq!(
                block_full_id(
                    block_at(&region, [0, 1, 0]).expect("snow should support scaffolding")
                ),
                "minecraft:scaffolding[bottom=false,distance=0,waterlogged=false]",
                "unexpected scaffolding support result for {snow_block}"
            );
        }
    }

    #[test]
    fn slab_over_soul_sand_applies_soul_sand_speed_factor() {
        let mut region = region_with_shape([4, 3, 1]);
        for x in 0..=3 {
            region
                .set_block([x, 0, 0], &parse_block("minecraft:soul_sand"))
                .unwrap();
            region
                .set_block(
                    [x, 1, 0],
                    &parse_block("minecraft:oak_slab[type=bottom,waterlogged=false]"),
                )
                .unwrap();
        }
        let world = LoadedSchematic {
            name: "slab-runway-soul-sand".to_string(),
            region,
            approximate_collision_blocks: Vec::new(),
        };
        let command = VerifyCommand {
            input: std::path::PathBuf::from("slab-runway-soul-sand.litematic"),
            out: std::path::PathBuf::from("artifacts/test"),
            target_speed: 0.0,
            ticks: 1,
            inspect_tick: Some(1),
            start_x: 0.5,
            start_y: 1.5,
            start_z: 0.5,
            start_vx: 1.0,
            start_vy: 0.0,
            start_vz: 0.0,
            start_on_ground: true,
            width: VERIFY_DEFAULT_WIDTH,
            height: VERIFY_DEFAULT_HEIGHT,
            entity_id_mod4: 0,
            initial_tick_count: 0,
            entity_rng_seed: None,
            entity_uuid: None,
            bootstrap_fluids: false,
            entity_kind: VerifyEntityKind::Item,
            no_ai: false,

            no_gravity: false,
            fire_immune: false,
            start_fire_ticks: 0,
            item_health: None,
        };

        let rows = simulate(&world, &command);
        let tick = &rows[1];
        let expected_vx = (0.4_f32 as f64) * (0.6_f32 as f64) * HORIZONTAL_MOVEMENT_DAMPING;
        assert!((tick.vx - expected_vx).abs() < 1.0e-12);
        assert_eq!(tick.support_block, "minecraft:soul_sand");
    }

    #[test]
    fn honey_slide_formula_matches_vanilla_strong_descent_branch() {
        let mut velocity = Vec3d::new(0.4, honey_new_delta_y(-0.2), 0.3);
        apply_honey_slide(&mut velocity);
        assert!((velocity.x - 0.1).abs() < 1.0e-12);
        assert!((velocity.y - honey_new_delta_y(HONEY_SLIDE_TARGET_OLD_DELTA_Y)).abs() < 1.0e-12);
        assert!((velocity.z - 0.075).abs() < 1.0e-12);
    }

    #[test]
    fn honey_slide_formula_matches_vanilla_mild_descent_branch() {
        let mut velocity = Vec3d::new(0.4, honey_new_delta_y(-0.1), 0.3);
        apply_honey_slide(&mut velocity);
        assert!((velocity.x - 0.4).abs() < 1.0e-12);
        assert!((velocity.y - honey_new_delta_y(HONEY_SLIDE_TARGET_OLD_DELTA_Y)).abs() < 1.0e-12);
        assert!((velocity.z - 0.3).abs() < 1.0e-12);
    }

    #[test]
    fn simulate_applies_honey_slide_inside_block_effect() {
        let mut region = region_with_shape([3, 3, 1]);
        region
            .set_block([1, 0, 0], &parse_block("minecraft:honey_block"))
            .unwrap();
        let world = LoadedSchematic {
            name: "honey-slide".to_string(),
            region,
            approximate_collision_blocks: Vec::new(),
        };
        let command = VerifyCommand {
            input: std::path::PathBuf::from("honey-slide.litematic"),
            out: std::path::PathBuf::from("artifacts/test"),
            target_speed: 0.0,
            ticks: 1,
            inspect_tick: Some(1),
            start_x: 2.07,
            start_y: 0.5,
            start_z: 0.5,
            start_vx: 0.0,
            start_vy: -0.2,
            start_vz: 0.3,
            start_on_ground: false,
            width: VERIFY_DEFAULT_WIDTH,
            height: VERIFY_DEFAULT_HEIGHT,
            entity_id_mod4: 0,
            initial_tick_count: 0,
            entity_rng_seed: None,
            entity_uuid: None,
            bootstrap_fluids: false,
            entity_kind: VerifyEntityKind::Item,
            no_ai: false,

            no_gravity: false,
            fire_immune: false,
            start_fire_ticks: 0,
            item_health: None,
        };

        let rows = simulate(&world, &command);
        let tick = &rows[1];
        let pre_honey_velocity_y = -0.2 - GRAVITY;
        let expected_vy =
            honey_new_delta_y(HONEY_SLIDE_TARGET_OLD_DELTA_Y) * VERTICAL_MOVEMENT_DAMPING;
        let expected_vz = 0.3
            * (HONEY_SLIDE_TARGET_OLD_DELTA_Y / honey_old_delta_y(pre_honey_velocity_y))
            * HORIZONTAL_MOVEMENT_DAMPING;
        assert!((tick.x - 2.07).abs() < 1.0e-12);
        assert!((tick.y - 0.26).abs() < 1.0e-12);
        assert!((tick.vx - 0.0).abs() < 1.0e-12);
        assert!((tick.vy - expected_vy).abs() < 1.0e-12);
        assert!((tick.vz - expected_vz).abs() < 1.0e-12);
        assert!(!tick.on_ground);
    }

    #[test]
    fn simulate_item_honey_slide_matches_vanilla_probe() {
        let mut region = region_with_shape([3, 3, 1]);
        region
            .set_block([1, 0, 0], &parse_block("minecraft:honey_block"))
            .unwrap();
        let world = LoadedSchematic {
            name: "honey-slide-vanilla".to_string(),
            region,
            approximate_collision_blocks: Vec::new(),
        };
        let command = VerifyCommand {
            input: std::path::PathBuf::from("honey-slide-vanilla.litematic"),
            out: std::path::PathBuf::from("artifacts/test"),
            target_speed: 0.0,
            ticks: 1,
            inspect_tick: Some(1),
            start_x: 2.07,
            start_y: 0.5,
            start_z: 0.5,
            start_vx: 0.0,
            start_vy: -0.2,
            start_vz: 0.3,
            start_on_ground: false,
            width: VERIFY_DEFAULT_WIDTH,
            height: VERIFY_DEFAULT_HEIGHT,
            entity_id_mod4: 0,
            initial_tick_count: 0,
            entity_rng_seed: None,
            entity_uuid: None,
            bootstrap_fluids: false,
            entity_kind: VerifyEntityKind::Item,
            no_ai: false,
            no_gravity: false,
            fire_immune: false,
            start_fire_ticks: 0,
            item_health: None,
        };

        let rows = simulate(&world, &command);
        let tick = &rows[1];
        assert!((tick.x - 2.07).abs() < 5.0e-8);
        assert!((tick.y - 0.26).abs() < 5.0e-8);
        assert!((tick.z - 0.8).abs() < 5.0e-8);
        assert!((tick.vx - 0.0).abs() < 5.0e-8);
        assert!((tick.vy - (-0.124_852_004_859_924_36)).abs() < 5.0e-8);
        assert!((tick.vz - 0.089_146_043_915_758_46).abs() < 5.0e-8);
        assert!(!tick.on_ground);
        assert!(tick.alive);
        assert!(tick.removed_by.is_empty());
    }

    #[test]
    fn bubble_column_counts_as_source_water() {
        let mut region = region_with_shape([1, 1, 1]);
        region
            .set_block(
                [0, 0, 0],
                &parse_block("minecraft:bubble_column[drag=false]"),
            )
            .unwrap();

        let fluid = water_at(&region, [0, 0, 0]).expect("bubble column should expose water");
        let expected_height = (8.0_f32 / 9.0_f32) as f64;
        assert!((fluid.height - expected_height).abs() < 1.0e-12);
        assert!((fluid.own_height - (8.0_f32 / 9.0_f32) as f64).abs() < 1.0e-12);
        assert!(!fluid.falling);
    }

    #[test]
    fn lava_cells_expose_source_and_falling_heights() {
        let mut region = region_with_shape([2, 1, 1]);
        region
            .set_block([0, 0, 0], &parse_block("minecraft:lava"))
            .unwrap();
        region
            .set_block([1, 0, 0], &parse_block("minecraft:lava[level=10]"))
            .unwrap();

        let source = lava_at(&region, [0, 0, 0]).expect("source lava");
        assert!((source.height - (8.0_f32 / 9.0_f32) as f64).abs() < 1.0e-12);
        assert!(!source.falling);

        let falling = lava_at(&region, [1, 0, 0]).expect("falling lava");
        assert!((falling.own_height - (6.0_f32 / 9.0_f32) as f64).abs() < 1.0e-12);
        assert!(falling.falling);
    }

    #[test]
    fn simulate_applies_lava_drag_and_buoyancy_like_item_entity() {
        let mut region = region_with_shape([1, 2, 1]);
        region
            .set_block([0, 0, 0], &parse_block("minecraft:lava"))
            .unwrap();
        let world = LoadedSchematic {
            name: "lava-column".to_string(),
            region,
            approximate_collision_blocks: Vec::new(),
        };
        let command = VerifyCommand {
            input: std::path::PathBuf::from("lava-column.litematic"),
            out: std::path::PathBuf::from("artifacts/test"),
            target_speed: 0.0,
            ticks: 1,
            inspect_tick: Some(1),
            start_x: 0.5,
            start_y: 0.0,
            start_z: 0.5,
            start_vx: 0.4,
            start_vy: 0.0,
            start_vz: 0.2,
            start_on_ground: false,
            width: VERIFY_DEFAULT_WIDTH,
            height: VERIFY_DEFAULT_HEIGHT,
            entity_id_mod4: 0,
            initial_tick_count: 0,
            entity_rng_seed: None,
            entity_uuid: None,
            bootstrap_fluids: false,
            entity_kind: VerifyEntityKind::Item,
            no_ai: false,

            no_gravity: false,
            fire_immune: false,
            start_fire_ticks: 0,
            item_health: None,
        };

        let rows = simulate(&world, &command);
        let tick = &rows[1];
        assert!(!tick.in_water);
        assert!(tick.in_lava);
        assert_eq!(tick.active_fluid, "lava");
        assert_eq!(tick.item_health, Some(1));
        assert_eq!(tick.remaining_fire_ticks, 300);
        assert!((tick.y - BUOYANCY).abs() < 1.0e-12);
        assert!(
            (tick.vx - 0.4 * HORIZONTAL_LAVA_DAMPING * HORIZONTAL_MOVEMENT_DAMPING).abs() < 1.0e-12
        );
        assert!(
            (tick.vz - 0.2 * HORIZONTAL_LAVA_DAMPING * HORIZONTAL_MOVEMENT_DAMPING).abs() < 1.0e-12
        );
        assert!((tick.vy - BUOYANCY * VERTICAL_MOVEMENT_DAMPING).abs() < 1.0e-12);
    }

    #[test]
    fn simulate_supported_fire_matches_vanilla_one_tick_probe() {
        let mut region = region_with_shape([1, 2, 1]);
        region
            .set_block([0, 0, 0], &parse_block("minecraft:netherrack"))
            .unwrap();
        region
            .set_block([0, 1, 0], &parse_block("minecraft:fire"))
            .unwrap();
        let world = LoadedSchematic {
            name: "fire-supported".to_string(),
            region,
            approximate_collision_blocks: Vec::new(),
        };
        let command = VerifyCommand {
            input: std::path::PathBuf::from("fire-supported.litematic"),
            out: std::path::PathBuf::from("artifacts/test"),
            target_speed: 0.0,
            ticks: 1,
            inspect_tick: Some(1),
            start_x: 0.5,
            start_y: 1.0,
            start_z: 0.5,
            start_vx: 0.0,
            start_vy: 0.0,
            start_vz: 0.0,
            start_on_ground: false,
            width: VERIFY_DEFAULT_WIDTH,
            height: VERIFY_DEFAULT_HEIGHT,
            entity_id_mod4: 0,
            initial_tick_count: 0,
            entity_rng_seed: None,
            entity_uuid: None,
            bootstrap_fluids: false,
            entity_kind: VerifyEntityKind::Item,
            no_ai: false,

            no_gravity: false,
            fire_immune: false,
            start_fire_ticks: 0,
            item_health: None,
        };

        let rows = simulate(&world, &command);
        let tick = &rows[1];
        assert!(tick.alive);
        assert_eq!(tick.item_health, Some(4));
        assert_eq!(tick.remaining_fire_ticks, 160);
        assert!(tick.on_ground);
        assert!((tick.x - 0.5).abs() < 1.0e-12);
        assert!((tick.y - 1.0).abs() < 1.0e-12);
        assert!(tick.vx.abs() < 1.0e-12);
        assert!(tick.vy.abs() < 1.0e-12);
        assert!(tick.vz.abs() < 1.0e-12);
    }

    #[test]
    fn lava_inside_effect_ignites_and_destroys_default_item_on_second_tick() {
        let mut region = region_with_shape([1, 2, 1]);
        region
            .set_block([0, 0, 0], &parse_block("minecraft:lava"))
            .unwrap();
        let world = LoadedSchematic {
            name: "lava-damage".to_string(),
            region,
            approximate_collision_blocks: Vec::new(),
        };
        let command = VerifyCommand {
            input: std::path::PathBuf::from("lava-damage.litematic"),
            out: std::path::PathBuf::from("artifacts/test"),
            target_speed: 0.0,
            ticks: 2,
            inspect_tick: Some(2),
            start_x: 0.5,
            start_y: 0.0,
            start_z: 0.5,
            start_vx: 0.0,
            start_vy: 0.0,
            start_vz: 0.0,
            start_on_ground: false,
            width: VERIFY_DEFAULT_WIDTH,
            height: VERIFY_DEFAULT_HEIGHT,
            entity_id_mod4: 0,
            initial_tick_count: 0,
            entity_rng_seed: None,
            entity_uuid: None,
            bootstrap_fluids: false,
            entity_kind: VerifyEntityKind::Item,
            no_ai: false,

            no_gravity: false,
            fire_immune: false,
            start_fire_ticks: 0,
            item_health: None,
        };

        let rows = simulate(&world, &command);
        assert!(rows[1].alive);
        assert!(rows[1].on_fire);
        assert_eq!(rows[1].remaining_fire_ticks, 300);
        assert_eq!(rows[1].item_health, Some(1));
        assert!(!rows[2].alive);
        assert_eq!(rows[2].removed_by, "lavaHurt");
        assert_eq!(rows[2].item_health, Some(-3));
    }

    #[test]
    fn water_inside_effect_clears_fire_without_extra_damage_when_ticks_are_not_mod20() {
        let mut region = region_with_shape([1, 2, 1]);
        region
            .set_block([0, 0, 0], &parse_block("minecraft:water"))
            .unwrap();
        let world = LoadedSchematic {
            name: "water-extinguish".to_string(),
            region,
            approximate_collision_blocks: Vec::new(),
        };
        let command = VerifyCommand {
            input: std::path::PathBuf::from("water-extinguish.litematic"),
            out: std::path::PathBuf::from("artifacts/test"),
            target_speed: 0.0,
            ticks: 1,
            inspect_tick: Some(1),
            start_x: 0.5,
            start_y: 0.0,
            start_z: 0.5,
            start_vx: 0.0,
            start_vy: 0.0,
            start_vz: 0.0,
            start_on_ground: false,
            width: VERIFY_DEFAULT_WIDTH,
            height: VERIFY_DEFAULT_HEIGHT,
            entity_id_mod4: 0,
            initial_tick_count: 0,
            entity_rng_seed: None,
            entity_uuid: None,
            bootstrap_fluids: false,
            entity_kind: VerifyEntityKind::Item,
            no_ai: false,

            no_gravity: false,
            fire_immune: false,
            start_fire_ticks: 39,
            item_health: None,
        };

        let rows = simulate(&world, &command);
        assert!(rows[1].alive);
        assert!(!rows[1].on_fire);
        assert_eq!(rows[1].remaining_fire_ticks, 0);
        assert_eq!(rows[1].item_health, Some(5));
    }

    #[test]
    fn fire_and_soul_fire_apply_vanilla_ignite_and_damage_steps() {
        for (block_id, expected_health) in [("minecraft:fire", 4), ("minecraft:soul_fire", 3)] {
            let mut region = region_with_shape([1, 2, 1]);
            region.set_block([0, 0, 0], &parse_block(block_id)).unwrap();
            let world = LoadedSchematic {
                name: format!("fire-{block_id}"),
                region,
                approximate_collision_blocks: Vec::new(),
            };
            let command = VerifyCommand {
                input: std::path::PathBuf::from("fire-block.litematic"),
                out: std::path::PathBuf::from("artifacts/test"),
                target_speed: 0.0,
                ticks: 1,
                inspect_tick: Some(1),
                start_x: 0.5,
                start_y: 0.0,
                start_z: 0.5,
                start_vx: 0.0,
                start_vy: 0.0,
                start_vz: 0.0,
                start_on_ground: false,
                width: VERIFY_DEFAULT_WIDTH,
                height: VERIFY_DEFAULT_HEIGHT,
                entity_id_mod4: 0,
                initial_tick_count: 0,
                entity_rng_seed: None,
                entity_uuid: None,
                bootstrap_fluids: false,
                entity_kind: VerifyEntityKind::Item,
                no_ai: false,

                no_gravity: false,
                fire_immune: false,
                start_fire_ticks: 0,
                item_health: None,
            };

            let rows = simulate(&world, &command);
            assert!(
                rows[1].alive,
                "fire contact should not destroy item in one tick"
            );
            assert!(rows[1].on_fire);
            assert_eq!(rows[1].remaining_fire_ticks, 160);
            assert_eq!(
                rows[1].item_health,
                Some(expected_health),
                "unexpected item health for {block_id}"
            );
        }
    }

    #[test]
    fn fire_ignite_matches_vanilla_negative_tick_transition() {
        let mut deeply_negative = -5;
        fire_ignite(&mut deeply_negative, false);
        assert_eq!(deeply_negative, -4);

        let mut one_tick_away = -1;
        fire_ignite(&mut one_tick_away, false);
        assert_eq!(one_tick_away, 160);
    }

    #[test]
    fn lava_cauldron_inside_effect_matches_lava_ignite_and_hurt() {
        let mut region = region_with_shape([1, 1, 1]);
        region
            .set_block([0, 0, 0], &parse_block("minecraft:lava_cauldron"))
            .unwrap();
        let mut velocity = Vec3d::ZERO;
        let mut stuck_speed_multiplier = Vec3d::ZERO;
        let mut fall_distance = 0.0_f64;
        let mut remaining_fire_ticks = 0;
        let mut item_health = default_item_health(VerifyEntityKind::Item);

        let removed = apply_inside_block_effects(
            &region,
            Vec3d::new(0.5, 0.5, 0.5),
            Vec3d::new(0.5, 0.5, 0.5),
            VERIFY_DEFAULT_WIDTH,
            VERIFY_DEFAULT_HEIGHT,
            VerifyEntityKind::Item,
            false,
            false,
            &mut velocity,
            &mut stuck_speed_multiplier,
            &mut fall_distance,
            &mut remaining_fire_ticks,
            &mut item_health,
        );

        assert!(removed.is_none());
        assert_eq!(remaining_fire_ticks, 300);
        assert_eq!(item_health, Some(1));
    }

    #[test]
    fn cactus_inside_effect_damages_item_entities() {
        let mut region = region_with_shape([1, 1, 1]);
        region
            .set_block([0, 0, 0], &parse_block("minecraft:cactus[age=0]"))
            .unwrap();
        let mut velocity = Vec3d::ZERO;
        let mut stuck_speed_multiplier = Vec3d::ZERO;
        let mut fall_distance = 0.0_f64;
        let mut remaining_fire_ticks = 0;
        let mut item_health = default_item_health(VerifyEntityKind::Item);

        let removed = apply_inside_block_effects(
            &region,
            Vec3d::new(0.5, 0.5, 0.5),
            Vec3d::new(0.5, 0.5, 0.5),
            VERIFY_DEFAULT_WIDTH,
            VERIFY_DEFAULT_HEIGHT,
            VerifyEntityKind::Item,
            false,
            false,
            &mut velocity,
            &mut stuck_speed_multiplier,
            &mut fall_distance,
            &mut remaining_fire_ticks,
            &mut item_health,
        );

        assert!(removed.is_none());
        assert_eq!(item_health, Some(4));
        assert_eq!(remaining_fire_ticks, 0);
    }

    #[test]
    fn lit_campfires_damage_living_entities_without_igniting_them() {
        for (block_id, expected_health) in [
            (
                "minecraft:campfire[lit=true,signal_fire=false,waterlogged=false,facing=north]",
                9,
            ),
            (
                "minecraft:soul_campfire[lit=true,signal_fire=false,waterlogged=false,facing=north]",
                8,
            ),
        ] {
            let mut region = region_with_shape([1, 1, 1]);
            region.set_block([0, 0, 0], &parse_block(block_id)).unwrap();
            let mut velocity = Vec3d::ZERO;
            let mut stuck_speed_multiplier = Vec3d::ZERO;
            let mut fall_distance = 0.0_f64;
            let mut remaining_fire_ticks = 0;
            let mut item_health = Some(10);

            let removed = apply_inside_block_effects(
                &region,
                Vec3d::new(0.5, 0.2, 0.5),
                Vec3d::new(0.5, 0.2, 0.5),
                VERIFY_DEFAULT_WIDTH,
                VERIFY_DEFAULT_HEIGHT,
                VerifyEntityKind::Living,
                false,
                false,
                &mut velocity,
                &mut stuck_speed_multiplier,
                &mut fall_distance,
                &mut remaining_fire_ticks,
                &mut item_health,
            );

            assert!(removed.is_none());
            assert_eq!(item_health, Some(expected_health));
            assert_eq!(remaining_fire_ticks, 0);
        }
    }

    #[test]
    fn moving_living_on_magma_block_takes_hot_floor_damage() {
        let mut region = region_with_shape([2, 1, 1]);
        for x in 0..=1 {
            region
                .set_block([x, 0, 0], &parse_block("minecraft:magma_block"))
                .unwrap();
        }
        let world = LoadedSchematic {
            name: "magma-step-living".to_string(),
            region,
            approximate_collision_blocks: Vec::new(),
        };
        let command = VerifyCommand {
            input: std::path::PathBuf::from("magma-step-living.litematic"),
            out: std::path::PathBuf::from("artifacts/test"),
            target_speed: 0.0,
            ticks: 1,
            inspect_tick: Some(1),
            start_x: 0.5,
            start_y: 1.0,
            start_z: 0.5,
            start_vx: 0.01,
            start_vy: 0.0,
            start_vz: 0.0,
            start_on_ground: true,
            width: VERIFY_DEFAULT_WIDTH,
            height: VERIFY_DEFAULT_HEIGHT,
            entity_id_mod4: 0,
            initial_tick_count: 0,
            entity_rng_seed: None,
            entity_uuid: None,
            bootstrap_fluids: false,
            entity_kind: VerifyEntityKind::Living,
            no_ai: false,
            no_gravity: false,
            fire_immune: false,
            start_fire_ticks: 0,
            item_health: Some(10),
        };

        let rows = simulate(&world, &command);
        assert_eq!(rows[1].health, Some(9));
        assert_eq!(rows[1].item_health, Some(9));
        assert!(rows[1].alive);
        assert!(rows[1].removed_by.is_empty());
    }

    #[test]
    fn simulate_armor_stand_magma_probe_matches_vanilla_first_tick() {
        let world = armor_stand_magma_probe_world();
        let mut command = armor_stand_verify_command("living-magma-step.litematic");
        command.start_x = 1.5;
        command.start_y = 64.0;
        command.start_z = 1.5;
        command.start_vx = 0.01;

        let rows = simulate(&world, &command);
        let tick = &rows[1];
        assert!((tick.x - 1.51).abs() < 1.0e-12);
        assert!((tick.y - 64.0).abs() < 1.0e-12);
        assert!((tick.z - 1.5).abs() < 1.0e-12);
        assert!((tick.vx - 0.009100000262260438).abs() < 1.0e-12);
        assert!((tick.vy + 0.0784000015258789).abs() < 1.0e-12);
        assert!((tick.vz - 0.0).abs() < 1.0e-12);
        assert_eq!(tick.health, Some(10));
        assert_eq!(tick.item_health, Some(10));
        assert!(!tick.on_ground);
        assert!(!tick.collided_x);
        assert!(!tick.collided_y);
        assert!(!tick.collided_z);
        assert!((tick.fall_distance - 0.0).abs() < 1.0e-12);
    }

    #[test]
    fn simulate_armor_stand_ladder_probe_matches_vanilla_first_tick() {
        let world = armor_stand_ladder_probe_world();
        let mut command = armor_stand_verify_command("living-ladder-passive-step.litematic");
        command.start_x = 1.5;
        command.start_y = 65.2;
        command.start_z = 1.5;
        command.start_vy = -1.0;
        command.item_health = Some(20);

        let rows = simulate(&world, &command);
        let tick = &rows[1];
        assert!((tick.x - 1.5).abs() < 1.0e-12);
        assert!((tick.y - 65.04999999403954).abs() < 1.0e-12);
        assert!((tick.z - 1.5).abs() < 1.0e-12);
        assert!((tick.vx - 0.0).abs() < 1.0e-12);
        assert!((tick.vy + 0.22540001022815717).abs() < 1.0e-12);
        assert!((tick.vz - 0.0).abs() < 1.0e-12);
        assert_eq!(tick.health, Some(20));
        assert_eq!(tick.item_health, Some(20));
        assert!(!tick.on_ground);
        assert!((tick.fall_distance - 0.15000000596046448).abs() < 1.0e-12);
    }

    #[test]
    fn simulate_armor_stand_trapdoor_ladder_probe_matches_vanilla_first_tick() {
        let world = armor_stand_trapdoor_ladder_probe_world();
        let mut command =
            armor_stand_verify_command("living-trapdoor-ladder-passive-step.litematic");
        command.start_x = 1.5;
        command.start_y = 65.2;
        command.start_z = 1.5;
        command.start_vy = -1.0;
        command.item_health = Some(20);

        let rows = simulate(&world, &command);
        let tick = &rows[1];
        assert!((tick.x - 1.5).abs() < 1.0e-12);
        assert!((tick.y - 65.04999999403954).abs() < 1.0e-12);
        assert!((tick.z - 1.5).abs() < 1.0e-12);
        assert!((tick.vx - 0.0).abs() < 1.0e-12);
        assert!((tick.vy + 0.22540001022815717).abs() < 1.0e-12);
        assert!((tick.vz - 0.0).abs() < 1.0e-12);
        assert_eq!(tick.health, Some(20));
        assert_eq!(tick.item_health, Some(20));
        assert!(!tick.on_ground);
        assert!((tick.fall_distance - 0.15000000596046448).abs() < 1.0e-12);
    }

    #[test]
    fn simulate_armor_stand_ladder_horizontal_probe_matches_vanilla_first_tick() {
        let world = armor_stand_ladder_horizontal_probe_world();
        let mut command =
            armor_stand_verify_command("living-ladder-horizontal-collision-step.litematic");
        command.start_x = 1.75;
        command.start_y = 65.2;
        command.start_z = 1.5;
        command.start_vx = 1.0;
        command.item_health = Some(20);

        let rows = simulate(&world, &command);
        let tick = &rows[1];
        assert!((tick.x - 1.75).abs() < 1.0e-12);
        assert!((tick.y - 65.2).abs() < 1.0e-12);
        assert!((tick.z - 1.5).abs() < 1.0e-12);
        assert!((tick.vx - 0.0).abs() < 1.0e-12);
        assert!((tick.vy - 0.11760000228881837).abs() < 1.0e-12);
        assert!((tick.vz - 0.0).abs() < 1.0e-12);
        assert_eq!(tick.health, Some(20));
        assert_eq!(tick.item_health, Some(20));
        assert!(!tick.on_ground);
        assert!(tick.collided_x);
        assert!((tick.fall_distance - 0.0).abs() < 1.0e-12);
    }

    #[test]
    fn simulate_armor_stand_water_horizontal_probe_matches_vanilla_first_tick() {
        let world = armor_stand_water_horizontal_probe_world();
        let mut command =
            armor_stand_verify_command("living-water-horizontal-collision-step.litematic");
        command.start_x = 1.75;
        command.start_y = 64.0;
        command.start_z = 1.5;
        command.start_vx = 1.0;
        command.item_health = Some(20);

        let rows = simulate(&world, &command);
        let tick = &rows[1];
        assert!((tick.x - 1.75).abs() < 1.0e-12);
        assert!((tick.y - 64.0).abs() < 1.0e-12);
        assert!((tick.z - 1.5).abs() < 1.0e-12);
        assert!((tick.vx - 0.0).abs() < 1.0e-12);
        assert!((tick.vy + 0.005).abs() < 1.0e-12);
        assert!((tick.vz - 0.0).abs() < 1.0e-12);
        assert_eq!(tick.health, Some(20));
        assert_eq!(tick.item_health, Some(20));
        assert!(!tick.on_ground);
        assert!(tick.collided_x);
        assert!((tick.fall_distance - 0.0).abs() < 1.0e-12);
    }

    #[test]
    fn simulate_armor_stand_water_diagonal_wall_probe_matches_vanilla_first_tick() {
        let world = armor_stand_water_diagonal_wall_probe_world();
        let mut command =
            armor_stand_verify_command("living-water-diagonal-wall-jumpout-step.litematic");
        command.start_x = 1.75;
        command.start_y = 64.0;
        command.start_z = 1.75;
        command.start_vx = 1.0;
        command.start_vz = 1.0;
        command.item_health = Some(20);

        let rows = simulate(&world, &command);
        let tick = &rows[1];
        assert!((tick.x - 2.75).abs() < 1.0e-12);
        assert!((tick.y - 64.0).abs() < 1.0e-12);
        assert!((tick.z - 1.75).abs() < 1.0e-12);
        assert!((tick.vx - 0.800000011920929).abs() < 1.0e-12);
        assert!((tick.vy - 0.30000001192092896).abs() < 1.0e-12);
        assert!((tick.vz - 0.0).abs() < 1.0e-12);
        assert_eq!(tick.health, Some(20));
        assert_eq!(tick.item_health, Some(20));
        assert!(!tick.on_ground);
        assert!(tick.collided_z);
        assert!((tick.fall_distance - 0.0).abs() < 1.0e-12);
    }

    #[test]
    fn simulate_item_bamboo_horizontal_probe_matches_vanilla_first_tick() {
        let world = item_bamboo_horizontal_probe_world();
        let command = VerifyCommand {
            input: std::path::PathBuf::from("item-bamboo-horizontal-collision-two-step.litematic"),
            out: std::path::PathBuf::from("artifacts/test"),
            target_speed: 0.0,
            ticks: 1,
            inspect_tick: Some(1),
            start_x: 0.75,
            start_y: 65.2,
            start_z: 1.5,
            start_vx: 1.0,
            start_vy: 0.0,
            start_vz: 0.0,
            start_on_ground: false,
            width: VERIFY_DEFAULT_WIDTH,
            height: VERIFY_DEFAULT_HEIGHT,
            entity_id_mod4: 0,
            initial_tick_count: 0,
            entity_rng_seed: None,
            entity_uuid: None,
            bootstrap_fluids: false,
            entity_kind: VerifyEntityKind::Item,
            no_ai: false,
            no_gravity: true,
            fire_immune: false,
            start_fire_ticks: 0,
            item_health: Some(5),
        };

        let rows = simulate(&world, &command);
        let tick = &rows[1];
        assert!(
            (tick.x - 1.1979166716337204).abs() < 1.0e-12,
            "unexpected x={} vx={} y={} z={}",
            tick.x,
            tick.vx,
            tick.y,
            tick.z
        );
        assert!((tick.y - 65.2).abs() < 1.0e-12, "unexpected y={}", tick.y);
        assert!((tick.z - 1.5).abs() < 1.0e-12, "unexpected z={}", tick.z);
        assert!((tick.vx - 0.0).abs() < 1.0e-12, "unexpected vx={}", tick.vx);
        assert!((tick.vy - 0.0).abs() < 1.0e-12, "unexpected vy={}", tick.vy);
        assert!((tick.vz - 0.0).abs() < 1.0e-12, "unexpected vz={}", tick.vz);
        assert!(tick.collided_x);
        assert!(!tick.collided_y);
        assert!(!tick.collided_z);
        assert!(!tick.on_ground);
        assert!((tick.fall_distance - 0.0).abs() < 1.0e-12);
    }

    #[test]
    fn simulate_item_reloaded_unsupported_bamboo_snapshot_matches_vanilla() {
        let world = item_unsupported_bamboo_snapshot_world();
        let command = VerifyCommand {
            input: std::path::PathBuf::from("item-unsupported-bamboo-snapshot.litematic"),
            out: std::path::PathBuf::from("artifacts/test"),
            target_speed: 0.0,
            ticks: 2,
            inspect_tick: Some(2),
            start_x: 0.75,
            start_y: 65.2,
            start_z: 1.5,
            start_vx: 1.0,
            start_vy: 0.0,
            start_vz: 0.0,
            start_on_ground: false,
            width: VERIFY_DEFAULT_WIDTH,
            height: VERIFY_DEFAULT_HEIGHT,
            entity_id_mod4: 0,
            initial_tick_count: 0,
            entity_rng_seed: None,
            entity_uuid: None,
            bootstrap_fluids: false,
            entity_kind: VerifyEntityKind::Item,
            no_ai: false,
            no_gravity: true,
            fire_immune: false,
            start_fire_ticks: 0,
            item_health: Some(5),
        };

        let rows = simulate(&world, &command);
        let tick1 = &rows[1];
        assert!(
            (tick1.x - 1.75).abs() < 1.0e-12,
            "unexpected tick1 x={}",
            tick1.x
        );
        assert!(
            (tick1.vx - 0.9800000190734863).abs() < 1.0e-12,
            "unexpected tick1 vx={}",
            tick1.vx
        );
        assert!(
            (tick1.y - 65.2).abs() < 1.0e-12,
            "unexpected tick1 y={}",
            tick1.y
        );
        assert!(
            (tick1.z - 1.5).abs() < 1.0e-12,
            "unexpected tick1 z={}",
            tick1.z
        );
        assert!(!tick1.collided_x);
        assert_eq!(tick1.center_block, "minecraft:air");

        let tick2 = &rows[2];
        assert!(
            (tick2.x - 2.7300000190734863).abs() < 1.0e-12,
            "unexpected tick2 x={}",
            tick2.x
        );
        assert!(
            (tick2.vx - 0.9604000373840336).abs() < 1.0e-12,
            "unexpected tick2 vx={}",
            tick2.vx
        );
        assert!(
            (tick2.y - 65.2).abs() < 1.0e-12,
            "unexpected tick2 y={}",
            tick2.y
        );
        assert!(
            (tick2.z - 1.5).abs() < 1.0e-12,
            "unexpected tick2 z={}",
            tick2.z
        );
        assert!(!tick2.collided_x);
        assert_eq!(tick2.center_block, "minecraft:air");
    }

    #[test]
    fn simulate_item_reloaded_unsupported_scaffolding_snapshot_matches_vanilla() {
        let world = item_unsupported_scaffolding_snapshot_world();
        let command = VerifyCommand {
            input: std::path::PathBuf::from("item-unsupported-scaffolding-snapshot.litematic"),
            out: std::path::PathBuf::from("artifacts/test"),
            target_speed: 0.0,
            ticks: 2,
            inspect_tick: Some(2),
            start_x: 0.75,
            start_y: 65.2,
            start_z: 1.5,
            start_vx: 1.0,
            start_vy: 0.0,
            start_vz: 0.0,
            start_on_ground: false,
            width: VERIFY_DEFAULT_WIDTH,
            height: VERIFY_DEFAULT_HEIGHT,
            entity_id_mod4: 0,
            initial_tick_count: 0,
            entity_rng_seed: None,
            entity_uuid: None,
            bootstrap_fluids: false,
            entity_kind: VerifyEntityKind::Item,
            no_ai: false,
            no_gravity: true,
            fire_immune: false,
            start_fire_ticks: 0,
            item_health: Some(5),
        };

        let rows = simulate(&world, &command);
        let tick1 = &rows[1];
        assert!(
            (tick1.x - 1.75).abs() < 1.0e-12,
            "unexpected tick1 x={}",
            tick1.x
        );
        assert!(
            (tick1.vx - 0.9800000190734863).abs() < 1.0e-12,
            "unexpected tick1 vx={}",
            tick1.vx
        );
        assert!(
            (tick1.y - 65.2).abs() < 1.0e-12,
            "unexpected tick1 y={}",
            tick1.y
        );
        assert!(
            (tick1.z - 1.5).abs() < 1.0e-12,
            "unexpected tick1 z={}",
            tick1.z
        );
        assert!(!tick1.collided_x);
        assert_eq!(tick1.center_block, "minecraft:air");

        let tick2 = &rows[2];
        assert!(
            (tick2.x - 2.7300000190734863).abs() < 1.0e-12,
            "unexpected tick2 x={}",
            tick2.x
        );
        assert!(
            (tick2.vx - 0.9604000373840336).abs() < 1.0e-12,
            "unexpected tick2 vx={}",
            tick2.vx
        );
        assert!(
            (tick2.y - 65.2).abs() < 1.0e-12,
            "unexpected tick2 y={}",
            tick2.y
        );
        assert!(
            (tick2.z - 1.5).abs() < 1.0e-12,
            "unexpected tick2 z={}",
            tick2.z
        );
        assert!(!tick2.collided_x);
        assert_eq!(tick2.center_block, "minecraft:air");
    }

    #[test]
    fn stationary_living_on_magma_block_does_not_take_step_on_damage() {
        let mut region = region_with_shape([1, 1, 1]);
        region
            .set_block([0, 0, 0], &parse_block("minecraft:magma_block"))
            .unwrap();
        let world = LoadedSchematic {
            name: "magma-step-stationary".to_string(),
            region,
            approximate_collision_blocks: Vec::new(),
        };
        let command = VerifyCommand {
            input: std::path::PathBuf::from("magma-step-stationary.litematic"),
            out: std::path::PathBuf::from("artifacts/test"),
            target_speed: 0.0,
            ticks: 1,
            inspect_tick: Some(1),
            start_x: 0.5,
            start_y: 1.0,
            start_z: 0.5,
            start_vx: 0.0,
            start_vy: 0.0,
            start_vz: 0.0,
            start_on_ground: true,
            width: VERIFY_DEFAULT_WIDTH,
            height: VERIFY_DEFAULT_HEIGHT,
            entity_id_mod4: 0,
            initial_tick_count: 0,
            entity_rng_seed: None,
            entity_uuid: None,
            bootstrap_fluids: false,
            entity_kind: VerifyEntityKind::Living,
            no_ai: false,
            no_gravity: false,
            fire_immune: false,
            start_fire_ticks: 0,
            item_health: Some(10),
        };

        let rows = simulate(&world, &command);
        assert_eq!(rows[1].health, Some(10));
        assert!(rows[1].alive);
    }

    #[test]
    fn pointed_dripstone_tip_fall_damage_matches_guardian_formula() {
        assert_eq!(pointed_dripstone_fall_damage(0.5), 0);
        assert_eq!(pointed_dripstone_fall_damage(1.0), 1);
        assert_eq!(pointed_dripstone_fall_damage(2.0), 3);
    }

    #[test]
    fn landing_on_upward_pointed_dripstone_damages_living_entities() {
        let mut region = region_with_shape([1, 3, 1]);
        region
            .set_block([0, 0, 0], &parse_block("minecraft:dripstone_block"))
            .unwrap();
        region
            .set_block(
                [0, 1, 0],
                &parse_block(
                    "minecraft:pointed_dripstone[thickness=tip,vertical_direction=up,waterlogged=false]",
                ),
            )
            .unwrap();
        let world = LoadedSchematic {
            name: "pointed-dripstone-fall".to_string(),
            region,
            approximate_collision_blocks: Vec::new(),
        };
        let command = VerifyCommand {
            input: std::path::PathBuf::from("pointed-dripstone-fall.litematic"),
            out: std::path::PathBuf::from("artifacts/test"),
            target_speed: 0.0,
            ticks: 1,
            inspect_tick: Some(1),
            start_x: 0.5,
            start_y: 3.0,
            start_z: 0.5,
            start_vx: 0.0,
            start_vy: -1.5,
            start_vz: 0.0,
            start_on_ground: false,
            width: VERIFY_DEFAULT_WIDTH,
            height: VERIFY_DEFAULT_HEIGHT,
            entity_id_mod4: 0,
            initial_tick_count: 0,
            entity_rng_seed: None,
            entity_uuid: None,
            bootstrap_fluids: false,
            entity_kind: VerifyEntityKind::Living,
            no_ai: false,
            no_gravity: false,
            fire_immune: false,
            start_fire_ticks: 0,
            item_health: Some(10),
        };

        let rows = simulate(&world, &command);
        assert_eq!(rows[1].health, Some(9));
        assert!(rows[1].on_ground);
        assert!(rows[1].alive);
    }

    #[test]
    fn simulate_armor_stand_pointed_dripstone_probe_matches_vanilla_first_tick() {
        let world = armor_stand_pointed_dripstone_probe_world();
        let mut command =
            armor_stand_verify_command("living-pointed-dripstone-fall-step.litematic");
        command.start_x = 1.5;
        command.start_y = 65.0;
        command.start_z = 1.5;
        command.start_vy = -1.5;

        let rows = simulate(&world, &command);
        let tick = &rows[1];
        assert!((tick.x - 1.5).abs() < 1.0e-12);
        assert!(
            (tick.y - 63.5).abs() < 1.0e-12,
            "unexpected y={} vy={} on_ground={} collided_y={} fall_distance={}",
            tick.y,
            tick.vy,
            tick.on_ground,
            tick.collided_y,
            tick.fall_distance
        );
        assert!((tick.z - 1.5).abs() < 1.0e-12);
        assert!((tick.vx - 0.0).abs() < 1.0e-12);
        assert!((tick.vy + 1.5484000301361085).abs() < 1.0e-12);
        assert!((tick.vz - 0.0).abs() < 1.0e-12);
        assert_eq!(tick.health, Some(10));
        assert_eq!(tick.item_health, Some(10));
        assert!(!tick.on_ground);
        assert!(!tick.collided_y);
        assert!((tick.fall_distance - 1.5).abs() < 1.0e-12);
    }

    #[test]
    fn simulate_lit_campfire_matches_vanilla_living_probe() {
        let mut region = region_with_shape([1, 1, 1]);
        region
            .set_block(
                [0, 0, 0],
                &parse_block(
                    "minecraft:campfire[lit=true,signal_fire=false,waterlogged=false,facing=north]",
                ),
            )
            .unwrap();
        let world = LoadedSchematic {
            name: "campfire-living".to_string(),
            region,
            approximate_collision_blocks: Vec::new(),
        };
        let command = VerifyCommand {
            input: std::path::PathBuf::from("campfire-living.litematic"),
            out: std::path::PathBuf::from("artifacts/test"),
            target_speed: 0.0,
            ticks: 1,
            inspect_tick: Some(1),
            start_x: 0.5,
            start_y: 0.2,
            start_z: 0.5,
            start_vx: 0.0,
            start_vy: 0.0,
            start_vz: 0.0,
            start_on_ground: false,
            width: VERIFY_DEFAULT_WIDTH,
            height: VERIFY_DEFAULT_HEIGHT,
            entity_id_mod4: 0,
            initial_tick_count: 0,
            entity_rng_seed: None,
            entity_uuid: None,
            bootstrap_fluids: false,
            entity_kind: VerifyEntityKind::Living,
            no_ai: true,

            no_gravity: false,
            fire_immune: false,
            start_fire_ticks: 0,
            item_health: Some(10),
        };

        let rows = simulate(&world, &command);
        let tick = &rows[1];
        assert!(tick.alive);
        assert_eq!(tick.health, Some(9));
        assert_eq!(tick.item_health, Some(9));
        assert_eq!(tick.remaining_fire_ticks, 0);
        assert!((tick.x - 0.5).abs() < 1.0e-12);
        assert!((tick.y - 0.2).abs() < 1.0e-12);
        assert!(tick.vx.abs() < 1.0e-12, "unexpected tick: {:?}", tick);
        assert!(tick.vy.abs() < 1.0e-12, "unexpected tick: {:?}", tick);
        assert!(tick.vz.abs() < 1.0e-12, "unexpected tick: {:?}", tick);
    }

    #[test]
    fn no_ai_living_skips_travel_and_keeps_position() {
        let world = LoadedSchematic {
            name: "no-ai-living".to_string(),
            region: region_with_shape([1, 1, 1]),
            approximate_collision_blocks: Vec::new(),
        };
        let command = VerifyCommand {
            input: std::path::PathBuf::from("no-ai-living.litematic"),
            out: std::path::PathBuf::from("artifacts/test"),
            target_speed: 0.0,
            ticks: 1,
            inspect_tick: Some(1),
            start_x: 0.5,
            start_y: 0.5,
            start_z: 0.5,
            start_vx: 0.3,
            start_vy: -0.1,
            start_vz: 0.2,
            start_on_ground: false,
            width: VERIFY_DEFAULT_WIDTH,
            height: VERIFY_DEFAULT_HEIGHT,
            entity_id_mod4: 0,
            initial_tick_count: 0,
            entity_rng_seed: None,
            entity_uuid: None,
            bootstrap_fluids: false,
            entity_kind: VerifyEntityKind::Living,
            no_ai: true,

            no_gravity: false,
            fire_immune: false,
            start_fire_ticks: 0,
            item_health: Some(10),
        };

        let rows = simulate(&world, &command);
        let tick = &rows[1];
        assert!((tick.x - 0.5).abs() < 1.0e-12);
        assert!((tick.y - 0.5).abs() < 1.0e-12);
        assert!((tick.z - 0.5).abs() < 1.0e-12);
        assert!((tick.vx - 0.3).abs() < 1.0e-12);
        assert!((tick.vy + 0.1).abs() < 1.0e-12);
        assert!((tick.vz - 0.2).abs() < 1.0e-12);
        assert!(!tick.moved);
    }

    #[test]
    fn no_gravity_no_ai_living_matches_vanilla_static_probe() {
        let world = LoadedSchematic {
            name: "no-gravity-no-ai-living".to_string(),
            region: region_with_shape([1, 1, 1]),
            approximate_collision_blocks: Vec::new(),
        };
        let command = VerifyCommand {
            input: std::path::PathBuf::from("no-gravity-no-ai-living.litematic"),
            out: std::path::PathBuf::from("artifacts/test"),
            target_speed: 0.0,
            ticks: 1,
            inspect_tick: Some(1),
            start_x: 0.5,
            start_y: 2.0,
            start_z: 0.5,
            start_vx: 0.0,
            start_vy: 0.0,
            start_vz: 0.0,
            start_on_ground: false,
            width: VERIFY_DEFAULT_WIDTH,
            height: VERIFY_DEFAULT_HEIGHT,
            entity_id_mod4: 0,
            initial_tick_count: 0,
            entity_rng_seed: None,
            entity_uuid: None,
            bootstrap_fluids: false,
            entity_kind: VerifyEntityKind::Living,
            no_ai: true,
            no_gravity: true,
            fire_immune: false,
            start_fire_ticks: 0,
            item_health: Some(10),
        };

        let rows = simulate(&world, &command);
        let tick = &rows[1];
        assert!((tick.x - 0.5).abs() < 1.0e-12);
        assert!((tick.y - 2.0).abs() < 1.0e-12);
        assert!((tick.z - 0.5).abs() < 1.0e-12);
        assert!(tick.vx.abs() < 1.0e-12);
        assert!(tick.vy.abs() < 1.0e-12);
        assert!(tick.vz.abs() < 1.0e-12);
        assert!(!tick.moved);
    }

    #[test]
    fn simulate_active_living_air_matches_vanilla_first_tick_probe() {
        let world = LoadedSchematic {
            name: "active-living-air".to_string(),
            region: region_with_shape([1, 1, 1]),
            approximate_collision_blocks: Vec::new(),
        };
        let command = VerifyCommand {
            input: std::path::PathBuf::from("active-living-air.litematic"),
            out: std::path::PathBuf::from("artifacts/test"),
            target_speed: 0.0,
            ticks: 1,
            inspect_tick: Some(1),
            start_x: 0.5,
            start_y: 2.0,
            start_z: 0.5,
            start_vx: 0.0,
            start_vy: 0.0,
            start_vz: 0.0,
            start_on_ground: false,
            width: 0.9,
            height: 0.9,
            entity_id_mod4: 0,
            initial_tick_count: 0,
            entity_rng_seed: None,
            entity_uuid: None,
            bootstrap_fluids: false,
            entity_kind: VerifyEntityKind::Living,
            no_ai: false,

            no_gravity: false,
            fire_immune: false,
            start_fire_ticks: 0,
            item_health: Some(10),
        };

        let rows = simulate(&world, &command);
        let tick = &rows[1];
        assert!(tick.alive);
        assert!((tick.x - 0.5).abs() < 1.0e-12);
        assert!((tick.y - 2.0).abs() < 1.0e-12);
        assert!((tick.z - 0.5).abs() < 1.0e-12);
        assert!(tick.vx.abs() < 1.0e-12);
        assert!((tick.vy + 0.0784000015258789).abs() < 1.0e-12);
        assert!(tick.vz.abs() < 1.0e-12);
        assert!(!tick.on_ground);
        assert!(!tick.moved);
    }

    #[test]
    fn simulate_active_living_air_matches_vanilla_second_tick_sequence() {
        let world = LoadedSchematic {
            name: "active-living-air-2".to_string(),
            region: region_with_shape([1, 1, 1]),
            approximate_collision_blocks: Vec::new(),
        };
        let command = VerifyCommand {
            input: std::path::PathBuf::from("active-living-air-2.litematic"),
            out: std::path::PathBuf::from("artifacts/test"),
            target_speed: 0.0,
            ticks: 2,
            inspect_tick: Some(2),
            start_x: 0.5,
            start_y: 2.0,
            start_z: 0.5,
            start_vx: 0.0,
            start_vy: 0.0,
            start_vz: 0.0,
            start_on_ground: false,
            width: 0.5,
            height: 1.975,
            entity_id_mod4: 0,
            initial_tick_count: 0,
            entity_rng_seed: None,
            entity_uuid: None,
            bootstrap_fluids: false,
            entity_kind: VerifyEntityKind::Living,
            no_ai: false,
            no_gravity: false,
            fire_immune: false,
            start_fire_ticks: 0,
            item_health: Some(20),
        };

        let rows = simulate(&world, &command);
        let tick = &rows[2];
        let drag = 0.98_f32 as f64;
        let tick1_vy = -0.08 * drag;
        let tick2_y = 2.0 + tick1_vy;
        let tick2_vy = (tick1_vy - 0.08) * drag;
        assert!(tick.alive);
        assert!((tick.x - 0.5).abs() < 1.0e-12);
        assert!((tick.y - tick2_y).abs() < 1.0e-12);
        assert!((tick.z - 0.5).abs() < 1.0e-12);
        assert!(tick.vx.abs() < 1.0e-12);
        assert!((tick.vy - tick2_vy).abs() < 1.0e-12);
        assert!(tick.vz.abs() < 1.0e-12);
        assert!(!tick.on_ground);
        assert!(tick.moved);
    }

    #[test]
    fn living_open_trapdoor_above_matching_ladder_counts_as_climbable() {
        let mut region = region_with_shape([1, 2, 1]);
        region
            .set_block([0, 0, 0], &parse_block("minecraft:ladder[facing=north]"))
            .unwrap();
        region
            .set_block(
                [0, 1, 0],
                &parse_block(
                    "minecraft:oak_trapdoor[facing=north,half=bottom,open=true,waterlogged=false]",
                ),
            )
            .unwrap();
        assert!(living_on_climbable(&region, Vec3d::new(0.5, 1.2, 0.5)));
    }

    #[test]
    fn living_open_trapdoor_requires_matching_ladder_facing() {
        let mut region = region_with_shape([1, 2, 1]);
        region
            .set_block([0, 0, 0], &parse_block("minecraft:ladder[facing=south]"))
            .unwrap();
        region
            .set_block(
                [0, 1, 0],
                &parse_block(
                    "minecraft:oak_trapdoor[facing=north,half=bottom,open=true,waterlogged=false]",
                ),
            )
            .unwrap();
        assert!(!living_on_climbable(&region, Vec3d::new(0.5, 1.2, 0.5)));
    }

    #[test]
    fn simulate_active_living_ladder_clamps_descent_and_resets_fall_distance() {
        let mut region = region_with_shape([1, 3, 1]);
        region
            .set_block([0, 1, 0], &parse_block("minecraft:ladder[facing=north]"))
            .unwrap();
        let world = LoadedSchematic {
            name: "active-living-ladder".to_string(),
            region,
            approximate_collision_blocks: Vec::new(),
        };
        let command = VerifyCommand {
            input: std::path::PathBuf::from("active-living-ladder.litematic"),
            out: std::path::PathBuf::from("artifacts/test"),
            target_speed: 0.0,
            ticks: 1,
            inspect_tick: Some(1),
            start_x: 0.5,
            start_y: 1.2,
            start_z: 0.5,
            start_vx: 0.0,
            start_vy: -1.0,
            start_vz: 0.0,
            start_on_ground: false,
            width: 0.5,
            height: 1.975,
            entity_id_mod4: 0,
            initial_tick_count: 0,
            entity_rng_seed: None,
            entity_uuid: None,
            bootstrap_fluids: false,
            entity_kind: VerifyEntityKind::Living,
            no_ai: false,
            no_gravity: false,
            fire_immune: false,
            start_fire_ticks: 0,
            item_health: Some(20),
        };

        let rows = simulate(&world, &command);
        let tick = &rows[1];
        let expected_delta_y = -LIVING_CLIMBABLE_MAX_DELTA;
        let expected_vy = (expected_delta_y - ENTITY_BASE_GRAVITY) * LIVING_VERTICAL_AIR_DRAG;
        assert!(tick.alive);
        assert!(
            (tick.y - (1.2 + expected_delta_y)).abs() < 1.0e-12,
            "unexpected y: {:?}",
            tick
        );
        assert!((tick.vy - expected_vy).abs() < 1.0e-12);
        assert!((tick.fall_distance - LIVING_CLIMBABLE_MAX_DELTA).abs() < 1.0e-12);
        assert!(tick.moved);
        assert!(!tick.on_ground);
    }

    #[test]
    fn simulate_active_living_open_trapdoor_above_ladder_uses_climbable_motion() {
        let mut region = region_with_shape([1, 3, 1]);
        region
            .set_block([0, 0, 0], &parse_block("minecraft:ladder[facing=north]"))
            .unwrap();
        region
            .set_block(
                [0, 1, 0],
                &parse_block(
                    "minecraft:oak_trapdoor[facing=north,half=bottom,open=true,waterlogged=false]",
                ),
            )
            .unwrap();
        let world = LoadedSchematic {
            name: "active-living-trapdoor-ladder".to_string(),
            region,
            approximate_collision_blocks: Vec::new(),
        };
        let command = VerifyCommand {
            input: std::path::PathBuf::from("active-living-trapdoor-ladder.litematic"),
            out: std::path::PathBuf::from("artifacts/test"),
            target_speed: 0.0,
            ticks: 1,
            inspect_tick: Some(1),
            start_x: 0.5,
            start_y: 1.2,
            start_z: 0.5,
            start_vx: 0.0,
            start_vy: -1.0,
            start_vz: 0.0,
            start_on_ground: false,
            width: 0.5,
            height: 1.975,
            entity_id_mod4: 0,
            initial_tick_count: 0,
            entity_rng_seed: None,
            entity_uuid: None,
            bootstrap_fluids: false,
            entity_kind: VerifyEntityKind::Living,
            no_ai: false,
            no_gravity: false,
            fire_immune: false,
            start_fire_ticks: 0,
            item_health: Some(20),
        };

        let rows = simulate(&world, &command);
        let tick = &rows[1];
        let expected_delta_y = -LIVING_CLIMBABLE_MAX_DELTA;
        let expected_vy = (expected_delta_y - ENTITY_BASE_GRAVITY) * LIVING_VERTICAL_AIR_DRAG;
        assert!(tick.alive);
        assert!(
            (tick.y - (1.2 + expected_delta_y)).abs() < 1.0e-12,
            "unexpected y: {:?}",
            tick
        );
        assert!(
            (tick.vy - expected_vy).abs() < 1.0e-12,
            "unexpected vy: {:?}",
            tick
        );
        assert!(
            (tick.fall_distance - LIVING_CLIMBABLE_MAX_DELTA).abs() < 1.0e-12,
            "unexpected fall distance: {:?}",
            tick
        );
        assert!(tick.moved);
        assert!(!tick.on_ground);
    }

    #[test]
    fn simulate_active_living_scaffolding_uses_climbable_motion() {
        let mut region = region_with_shape([1, 3, 1]);
        region
            .set_block([0, 0, 0], &parse_block("minecraft:stone"))
            .unwrap();
        region
            .set_block(
                [0, 1, 0],
                &parse_block("minecraft:scaffolding[bottom=false,distance=0,waterlogged=false]"),
            )
            .unwrap();
        let world = LoadedSchematic {
            name: "active-living-scaffolding".to_string(),
            region,
            approximate_collision_blocks: Vec::new(),
        };
        let command = VerifyCommand {
            input: std::path::PathBuf::from("active-living-scaffolding.litematic"),
            out: std::path::PathBuf::from("artifacts/test"),
            target_speed: 0.0,
            ticks: 1,
            inspect_tick: Some(1),
            start_x: 0.5,
            start_y: 1.2,
            start_z: 0.5,
            start_vx: 0.0,
            start_vy: -1.0,
            start_vz: 0.0,
            start_on_ground: false,
            width: 0.5,
            height: 1.975,
            entity_id_mod4: 0,
            initial_tick_count: 0,
            entity_rng_seed: None,
            entity_uuid: None,
            bootstrap_fluids: false,
            entity_kind: VerifyEntityKind::Living,
            no_ai: false,
            no_gravity: false,
            fire_immune: false,
            start_fire_ticks: 0,
            item_health: Some(20),
        };

        let rows = simulate(&world, &command);
        let tick = &rows[1];
        let expected_delta_y = -LIVING_CLIMBABLE_MAX_DELTA;
        let expected_vy = (expected_delta_y - ENTITY_BASE_GRAVITY) * LIVING_VERTICAL_AIR_DRAG;
        assert!(tick.alive);
        assert!((tick.y - (1.2 + expected_delta_y)).abs() < 1.0e-12);
        assert!((tick.vy - expected_vy).abs() < 1.0e-12);
        assert!((tick.fall_distance - LIVING_CLIMBABLE_MAX_DELTA).abs() < 1.0e-12);
        assert!(tick.moved);
        assert!(!tick.on_ground);
    }

    #[test]
    fn simulate_active_living_ladder_horizontal_collision_matches_vanilla_probe() {
        let mut region = region_with_shape([3, 3, 2]);
        region
            .set_block([1, 1, 0], &parse_block("minecraft:stone"))
            .unwrap();
        region
            .set_block([1, 1, 1], &parse_block("minecraft:ladder[facing=north]"))
            .unwrap();
        region
            .set_block([2, 1, 1], &parse_block("minecraft:stone"))
            .unwrap();
        let world = LoadedSchematic {
            name: "active-living-ladder-horizontal".to_string(),
            region,
            approximate_collision_blocks: Vec::new(),
        };
        let command = VerifyCommand {
            input: std::path::PathBuf::from("active-living-ladder-horizontal.litematic"),
            out: std::path::PathBuf::from("artifacts/test"),
            target_speed: 0.0,
            ticks: 1,
            inspect_tick: Some(1),
            start_x: 1.75,
            start_y: 1.2,
            start_z: 1.5,
            start_vx: 1.0,
            start_vy: 0.0,
            start_vz: 0.0,
            start_on_ground: false,
            width: 0.5,
            height: 1.975,
            entity_id_mod4: 0,
            initial_tick_count: 0,
            entity_rng_seed: None,
            entity_uuid: None,
            bootstrap_fluids: false,
            entity_kind: VerifyEntityKind::Living,
            no_ai: false,
            no_gravity: false,
            fire_immune: false,
            start_fire_ticks: 0,
            item_health: Some(20),
        };

        let rows = simulate(&world, &command);
        let tick = &rows[1];
        let expected_vy =
            (LIVING_CLIMBABLE_ASCENT - ENTITY_BASE_GRAVITY) * LIVING_VERTICAL_AIR_DRAG;
        assert!(tick.alive);
        assert!((tick.x - 1.75).abs() < 1.0e-12);
        assert!((tick.y - 1.2).abs() < 1.0e-12);
        assert!((tick.z - 1.5).abs() < 1.0e-12);
        assert!(tick.vx.abs() < 1.0e-12);
        assert!((tick.vy - expected_vy).abs() < 1.0e-12);
        assert!(tick.vz.abs() < 1.0e-12);
        assert!(tick.collided_x);
        assert!(!tick.on_ground);
        assert!((tick.fall_distance - 0.0).abs() < 1.0e-12);
    }

    #[test]
    fn simulate_active_living_water_horizontal_collision_stays_submerged_like_vanilla() {
        let mut region = region_with_shape([3, 1, 1]);
        region
            .set_block([1, 0, 0], &parse_block("minecraft:water"))
            .unwrap();
        region
            .set_block([2, 0, 0], &parse_block("minecraft:stone"))
            .unwrap();
        let world = LoadedSchematic {
            name: "active-living-water-horizontal".to_string(),
            region,
            approximate_collision_blocks: Vec::new(),
        };
        let command = VerifyCommand {
            input: std::path::PathBuf::from("active-living-water-horizontal.litematic"),
            out: std::path::PathBuf::from("artifacts/test"),
            target_speed: 0.0,
            ticks: 1,
            inspect_tick: Some(1),
            start_x: 1.75,
            start_y: 0.0,
            start_z: 0.5,
            start_vx: 1.0,
            start_vy: 0.0,
            start_vz: 0.0,
            start_on_ground: false,
            width: 0.5,
            height: 1.975,
            entity_id_mod4: 0,
            initial_tick_count: 0,
            entity_rng_seed: None,
            entity_uuid: None,
            bootstrap_fluids: false,
            entity_kind: VerifyEntityKind::Living,
            no_ai: false,
            no_gravity: false,
            fire_immune: false,
            start_fire_ticks: 0,
            item_health: Some(20),
        };

        let rows = simulate(&world, &command);
        let tick = &rows[1];
        assert!(tick.alive);
        assert!((tick.x - 1.75).abs() < 1.0e-12);
        assert!(tick.y.abs() < 1.0e-12);
        assert!((tick.z - 0.5).abs() < 1.0e-12);
        assert!(tick.vx.abs() < 1.0e-12);
        assert!((tick.vy + 0.005).abs() < 1.0e-12);
        assert!(tick.vz.abs() < 1.0e-12);
        assert!(tick.collided_x);
        assert!(!tick.moved);
        assert!((tick.fall_distance - 0.0).abs() < 1.0e-12);
    }

    #[test]
    fn simulate_active_living_water_diagonal_collision_jumps_out_like_vanilla() {
        let mut region = region_with_shape([5, 1, 4]);
        region
            .set_block([1, 0, 1], &parse_block("minecraft:water"))
            .unwrap();
        for x in 0..5 {
            region
                .set_block([x, 0, 2], &parse_block("minecraft:stone"))
                .unwrap();
        }
        let world = LoadedSchematic {
            name: "active-living-water-diagonal-jumpout".to_string(),
            region,
            approximate_collision_blocks: Vec::new(),
        };
        let command = VerifyCommand {
            input: std::path::PathBuf::from("active-living-water-diagonal-jumpout.litematic"),
            out: std::path::PathBuf::from("artifacts/test"),
            target_speed: 0.0,
            ticks: 1,
            inspect_tick: Some(1),
            start_x: 1.75,
            start_y: 0.0,
            start_z: 1.75,
            start_vx: 1.0,
            start_vy: 0.0,
            start_vz: 1.0,
            start_on_ground: false,
            width: 0.5,
            height: 1.975,
            entity_id_mod4: 0,
            initial_tick_count: 0,
            entity_rng_seed: None,
            entity_uuid: None,
            bootstrap_fluids: false,
            entity_kind: VerifyEntityKind::Living,
            no_ai: false,
            no_gravity: false,
            fire_immune: false,
            start_fire_ticks: 0,
            item_health: Some(20),
        };

        let rows = simulate(&world, &command);
        let tick = &rows[1];
        assert!(tick.alive);
        assert!((tick.x - 2.75).abs() < 1.0e-12);
        assert!(tick.y.abs() < 1.0e-12);
        assert!((tick.z - 1.75).abs() < 1.0e-12);
        assert!((tick.vx - (0.8_f32 as f64)).abs() < 1.0e-12);
        assert!((tick.vy - (0.3_f32 as f64)).abs() < 1.0e-12);
        assert!(tick.vz.abs() < 1.0e-12);
        assert!(!tick.collided_x);
        assert!(tick.collided_z);
        assert!(tick.moved);
        assert!((tick.fall_distance - 0.0).abs() < 1.0e-12);
    }

    #[test]
    fn simulate_active_living_lava_diagonal_collision_jumps_out_like_vanilla() {
        let mut region = region_with_shape([5, 1, 4]);
        region
            .set_block([1, 0, 1], &parse_block("minecraft:lava"))
            .unwrap();
        for x in 0..5 {
            region
                .set_block([x, 0, 2], &parse_block("minecraft:stone"))
                .unwrap();
        }
        let world = LoadedSchematic {
            name: "active-living-lava-diagonal-jumpout".to_string(),
            region,
            approximate_collision_blocks: Vec::new(),
        };
        let command = VerifyCommand {
            input: std::path::PathBuf::from("active-living-lava-diagonal-jumpout.litematic"),
            out: std::path::PathBuf::from("artifacts/test"),
            target_speed: 0.0,
            ticks: 1,
            inspect_tick: Some(1),
            start_x: 1.75,
            start_y: 0.0,
            start_z: 1.75,
            start_vx: 1.0,
            start_vy: 0.0,
            start_vz: 1.0,
            start_on_ground: false,
            width: 0.5,
            height: 1.975,
            entity_id_mod4: 0,
            initial_tick_count: 0,
            entity_rng_seed: None,
            entity_uuid: None,
            bootstrap_fluids: false,
            entity_kind: VerifyEntityKind::Living,
            no_ai: false,
            no_gravity: false,
            fire_immune: false,
            start_fire_ticks: 0,
            item_health: Some(20),
        };

        let rows = simulate(&world, &command);
        let tick = &rows[1];
        assert!(tick.alive);
        assert!((tick.x - 2.75).abs() < 1.0e-12);
        assert!(tick.y.abs() < 1.0e-12);
        assert!((tick.z - 1.75).abs() < 1.0e-12);
        assert!((tick.vx - 0.5).abs() < 1.0e-12);
        assert!((tick.vy - (0.3_f32 as f64)).abs() < 1.0e-12);
        assert!(tick.vz.abs() < 1.0e-12);
        assert!(!tick.collided_x);
        assert!(tick.collided_z);
        assert!(tick.moved);
        assert_eq!(tick.remaining_fire_ticks, 300);
        assert!((tick.fall_distance - 0.0).abs() < 1.0e-12);
    }

    #[test]
    fn simulate_active_living_water_matches_vanilla_passive_probe() {
        let mut region = region_with_shape([1, 1, 1]);
        region
            .set_block([0, 0, 0], &parse_block("minecraft:water"))
            .unwrap();
        let world = LoadedSchematic {
            name: "active-living-water".to_string(),
            region,
            approximate_collision_blocks: Vec::new(),
        };
        let command = VerifyCommand {
            input: std::path::PathBuf::from("active-living-water.litematic"),
            out: std::path::PathBuf::from("artifacts/test"),
            target_speed: 0.0,
            ticks: 1,
            inspect_tick: Some(1),
            start_x: 0.5,
            start_y: 0.0,
            start_z: 0.5,
            start_vx: 0.0,
            start_vy: 0.0,
            start_vz: 0.0,
            start_on_ground: false,
            width: 0.5,
            height: 1.975,
            entity_id_mod4: 0,
            initial_tick_count: 0,
            entity_rng_seed: None,
            entity_uuid: None,
            bootstrap_fluids: false,
            entity_kind: VerifyEntityKind::Living,
            no_ai: false,

            no_gravity: false,
            fire_immune: false,
            start_fire_ticks: 0,
            item_health: Some(20),
        };

        let rows = simulate(&world, &command);
        let tick = &rows[1];
        assert!(tick.alive);
        assert!((tick.x - 0.5).abs() < 1.0e-12);
        assert!((tick.y - 0.0).abs() < 1.0e-12);
        assert!((tick.z - 0.5).abs() < 1.0e-12);
        assert!(tick.vx.abs() < 1.0e-12);
        assert!((tick.vy + 0.005).abs() < 1.0e-12);
        assert!(tick.vz.abs() < 1.0e-12);
        assert!(!tick.on_ground);
        assert!(!tick.moved);
    }

    #[test]
    fn simulate_active_living_water_matches_vanilla_second_tick_sequence() {
        let mut region = region_with_shape([1, 1, 1]);
        region
            .set_block([0, 0, 0], &parse_block("minecraft:water"))
            .unwrap();
        let world = LoadedSchematic {
            name: "active-living-water-2".to_string(),
            region,
            approximate_collision_blocks: Vec::new(),
        };
        let command = VerifyCommand {
            input: std::path::PathBuf::from("active-living-water-2.litematic"),
            out: std::path::PathBuf::from("artifacts/test"),
            target_speed: 0.0,
            ticks: 2,
            inspect_tick: Some(2),
            start_x: 0.5,
            start_y: 0.0,
            start_z: 0.5,
            start_vx: 0.0,
            start_vy: 0.0,
            start_vz: 0.0,
            start_on_ground: false,
            width: 0.5,
            height: 1.975,
            entity_id_mod4: 0,
            initial_tick_count: 0,
            entity_rng_seed: None,
            entity_uuid: None,
            bootstrap_fluids: false,
            entity_kind: VerifyEntityKind::Living,
            no_ai: false,
            no_gravity: false,
            fire_immune: false,
            start_fire_ticks: 0,
            item_health: Some(20),
        };

        let rows = simulate(&world, &command);
        let tick = &rows[2];
        let fluid_drag = 0.8_f32 as f64;
        let tick1_vy = -0.08 / 16.0;
        let tick2_y = tick1_vy;
        let tick2_vy = tick1_vy * fluid_drag - 0.08 / 16.0;
        assert!(tick.alive);
        assert!((tick.x - 0.5).abs() < 1.0e-12);
        assert!((tick.y - tick2_y).abs() < 1.0e-12);
        assert!((tick.z - 0.5).abs() < 1.0e-12);
        assert!(tick.vx.abs() < 1.0e-12);
        assert!((tick.vy - tick2_vy).abs() < 1.0e-12);
        assert!(tick.vz.abs() < 1.0e-12);
        assert!(!tick.on_ground);
        assert!(tick.moved);
    }

    #[test]
    fn simulate_active_living_full_lava_matches_guardian_first_tick_sequence() {
        let mut region = region_with_shape([1, 1, 1]);
        region
            .set_block([0, 0, 0], &parse_block("minecraft:lava"))
            .unwrap();
        let world = LoadedSchematic {
            name: "active-living-full-lava".to_string(),
            region,
            approximate_collision_blocks: Vec::new(),
        };
        let command = VerifyCommand {
            input: std::path::PathBuf::from("active-living-full-lava.litematic"),
            out: std::path::PathBuf::from("artifacts/test"),
            target_speed: 0.0,
            ticks: 1,
            inspect_tick: Some(1),
            start_x: 0.5,
            start_y: 0.0,
            start_z: 0.5,
            start_vx: 0.0,
            start_vy: 0.0,
            start_vz: 0.0,
            start_on_ground: false,
            width: 0.5,
            height: 1.975,
            entity_id_mod4: 0,
            initial_tick_count: 0,
            entity_rng_seed: None,
            entity_uuid: None,
            bootstrap_fluids: false,
            entity_kind: VerifyEntityKind::Living,
            no_ai: false,
            no_gravity: false,
            fire_immune: true,
            start_fire_ticks: 0,
            item_health: None,
        };

        let rows = simulate(&world, &command);
        let tick = &rows[1];
        assert!(tick.alive);
        assert!((tick.x - 0.5).abs() < 1.0e-12);
        assert!((tick.y - 0.0).abs() < 1.0e-12);
        assert!((tick.z - 0.5).abs() < 1.0e-12);
        assert!(tick.vx.abs() < 1.0e-12);
        assert!((tick.vy + 0.02).abs() < 1.0e-12);
        assert!(tick.vz.abs() < 1.0e-12);
        assert!(!tick.on_ground);
        assert!(!tick.moved);
    }

    #[test]
    fn simulate_active_living_shallow_lava_matches_guardian_first_tick_sequence() {
        let mut region = region_with_shape([1, 1, 1]);
        region
            .set_block([0, 0, 0], &parse_block("minecraft:lava[level=7]"))
            .unwrap();
        let world = LoadedSchematic {
            name: "active-living-shallow-lava".to_string(),
            region,
            approximate_collision_blocks: Vec::new(),
        };
        let command = VerifyCommand {
            input: std::path::PathBuf::from("active-living-shallow-lava.litematic"),
            out: std::path::PathBuf::from("artifacts/test"),
            target_speed: 0.0,
            ticks: 1,
            inspect_tick: Some(1),
            start_x: 0.5,
            start_y: 0.0,
            start_z: 0.5,
            start_vx: 0.0,
            start_vy: 0.0,
            start_vz: 0.0,
            start_on_ground: false,
            width: 0.5,
            height: 1.975,
            entity_id_mod4: 0,
            initial_tick_count: 0,
            entity_rng_seed: None,
            entity_uuid: None,
            bootstrap_fluids: false,
            entity_kind: VerifyEntityKind::Living,
            no_ai: false,
            no_gravity: false,
            fire_immune: true,
            start_fire_ticks: 0,
            item_health: None,
        };

        let rows = simulate(&world, &command);
        let tick = &rows[1];
        assert!(tick.alive);
        assert!((tick.x - 0.5).abs() < 1.0e-12);
        assert!((tick.y - 0.0).abs() < 1.0e-12);
        assert!((tick.z - 0.5).abs() < 1.0e-12);
        assert!(tick.vx.abs() < 1.0e-12);
        assert!((tick.vy + 0.025).abs() < 1.0e-12);
        assert!(tick.vz.abs() < 1.0e-12);
        assert!(!tick.on_ground);
        assert!(!tick.moved);
    }

    #[test]
    fn source_water_above_bubble_support_persists_without_neighbor_event() {
        for support_block in ["minecraft:soul_sand", "minecraft:magma_block"] {
            let mut region = region_with_shape([1, 3, 1]);
            region
                .set_block([0, 0, 0], &parse_block(support_block))
                .unwrap();
            region
                .set_block([0, 1, 0], &parse_block("minecraft:water"))
                .unwrap();
            run_world_ticks(&mut region, 25);
            assert_eq!(
                block_full_id(block_at(&region, [0, 1, 0]).expect("snapshot water should persist")),
                "minecraft:water",
                "unexpected snapshot bootstrap bubble update for {support_block}"
            );
        }
    }

    #[test]
    fn scheduled_source_water_above_bubble_support_forms_column() {
        for (support_block, expected_column) in [
            ("minecraft:soul_sand", "minecraft:bubble_column[drag=false]"),
            (
                "minecraft:magma_block",
                "minecraft:bubble_column[drag=true]",
            ),
        ] {
            let mut region = region_with_shape([1, 3, 1]);
            region
                .set_block([0, 0, 0], &parse_block(support_block))
                .unwrap();
            region
                .set_block([0, 1, 0], &parse_block("minecraft:water"))
                .unwrap();
            let mut block_ticks = DynamicBlockTicks::default();
            block_ticks.schedule(BUBBLE_COLUMN_FORM_TICK_DELAY, [0, 1, 0]);
            block_ticks.run_due(&mut region, BUBBLE_COLUMN_FORM_TICK_DELAY);
            assert_eq!(
                block_full_id(block_at(&region, [0, 1, 0]).expect("bubble column should form")),
                expected_column,
                "unexpected bubble column state for {support_block}"
            );
        }
    }

    #[test]
    fn unsupported_bubble_column_persists_without_neighbor_event() {
        let mut region = region_with_shape([1, 2, 1]);
        region
            .set_block([0, 0, 0], &parse_block("minecraft:smooth_stone"))
            .unwrap();
        region
            .set_block(
                [0, 1, 0],
                &parse_block("minecraft:bubble_column[drag=true]"),
            )
            .unwrap();
        run_world_ticks(&mut region, 25);
        assert_eq!(
            block_full_id(
                block_at(&region, [0, 1, 0]).expect("snapshot bubble column should persist")
            ),
            "minecraft:bubble_column[drag=true]"
        );
    }

    #[test]
    fn scheduled_unsupported_bubble_column_reverts_to_source_water() {
        let mut region = region_with_shape([1, 2, 1]);
        region
            .set_block([0, 0, 0], &parse_block("minecraft:smooth_stone"))
            .unwrap();
        region
            .set_block(
                [0, 1, 0],
                &parse_block("minecraft:bubble_column[drag=true]"),
            )
            .unwrap();
        let mut block_ticks = DynamicBlockTicks::default();
        block_ticks.schedule(BUBBLE_COLUMN_CHECK_TICK_DELAY, [0, 1, 0]);
        block_ticks.run_due(&mut region, BUBBLE_COLUMN_CHECK_TICK_DELAY);
        assert_eq!(
            block_full_id(block_at(&region, [0, 1, 0]).expect("bubble column should revert")),
            "minecraft:water"
        );
    }

    #[test]
    fn invalid_bubble_column_is_inactive_for_item_motion_after_reload() {
        let mut region = region_with_shape([1, 2, 1]);
        region
            .set_block([0, 0, 0], &parse_block("minecraft:smooth_stone"))
            .unwrap();
        region
            .set_block(
                [0, 1, 0],
                &parse_block("minecraft:bubble_column[drag=true]"),
            )
            .unwrap();
        let world = LoadedSchematic {
            name: "invalid-bubble-column-snapshot".to_string(),
            region,
            approximate_collision_blocks: Vec::new(),
        };
        let command = VerifyCommand {
            input: std::path::PathBuf::from("invalid-bubble-column-snapshot.litematic"),
            out: std::path::PathBuf::from("artifacts/test"),
            target_speed: 0.0,
            ticks: 2,
            inspect_tick: Some(2),
            start_x: 0.5,
            start_y: 1.0,
            start_z: 0.5,
            start_vx: 0.0,
            start_vy: 0.0,
            start_vz: 0.0,
            start_on_ground: false,
            width: VERIFY_DEFAULT_WIDTH,
            height: VERIFY_DEFAULT_HEIGHT,
            entity_id_mod4: 0,
            initial_tick_count: 0,
            entity_rng_seed: None,
            entity_uuid: None,
            bootstrap_fluids: false,
            entity_kind: VerifyEntityKind::Item,
            no_ai: false,
            no_gravity: true,
            fire_immune: false,
            start_fire_ticks: 0,
            item_health: Some(5),
        };

        let rows = simulate(&world, &command);
        let tick1 = &rows[1];
        let tick2 = &rows[2];
        assert!((tick1.vy - 0.0004900000328104948).abs() < 1.0e-12);
        assert!((tick2.vy - 0.0009702000743107886).abs() < 1.0e-12);
        assert!(tick1.in_water);
        assert!(tick2.in_water);
    }

    #[test]
    fn valid_bubble_column_keeps_column_boost_after_reload() {
        let mut region = region_with_shape([1, 2, 1]);
        region
            .set_block([0, 0, 0], &parse_block("minecraft:soul_sand"))
            .unwrap();
        region
            .set_block(
                [0, 1, 0],
                &parse_block("minecraft:bubble_column[drag=false]"),
            )
            .unwrap();
        let world = LoadedSchematic {
            name: "valid-bubble-column-snapshot".to_string(),
            region,
            approximate_collision_blocks: Vec::new(),
        };
        let command = VerifyCommand {
            input: std::path::PathBuf::from("valid-bubble-column-snapshot.litematic"),
            out: std::path::PathBuf::from("artifacts/test"),
            target_speed: 0.0,
            ticks: 2,
            inspect_tick: Some(2),
            start_x: 0.5,
            start_y: 1.0,
            start_z: 0.5,
            start_vx: 0.0,
            start_vy: 0.0,
            start_vz: 0.0,
            start_on_ground: false,
            width: VERIFY_DEFAULT_WIDTH,
            height: VERIFY_DEFAULT_HEIGHT,
            entity_id_mod4: 0,
            initial_tick_count: 0,
            entity_rng_seed: None,
            entity_uuid: None,
            bootstrap_fluids: false,
            entity_kind: VerifyEntityKind::Item,
            no_ai: false,
            no_gravity: true,
            fire_immune: false,
            start_fire_ticks: 0,
            item_health: Some(5),
        };

        let rows = simulate(&world, &command);
        let tick1 = &rows[1];
        let tick2 = &rows[2];
        assert!((tick1.vy - 0.09849000194015914).abs() < 1.0e-12);
        assert!((tick2.vy - 0.1945202056872523).abs() < 1.0e-12);
        assert!(tick1.in_water);
        assert!(tick2.in_water);
    }

    #[test]
    fn bubble_column_velocity_formulas_match_vanilla() {
        let mut up_surface = Vec3d::new(0.0, 0.0005, 0.0);
        apply_above_bubble_column(&mut up_surface, false);
        assert!((up_surface.y - 0.1005).abs() < 1.0e-12);

        let mut up_inside = Vec3d::new(0.0, 0.0005, 0.0);
        apply_inside_bubble_column(&mut up_inside, false);
        assert!((up_inside.y - 0.0605).abs() < 1.0e-12);

        let mut down_surface = Vec3d::new(0.0, 0.0005, 0.0);
        apply_above_bubble_column(&mut down_surface, true);
        assert!((down_surface.y + 0.0295).abs() < 1.0e-12);
    }

    #[test]
    fn simulate_applies_bubble_column_surface_boost() {
        let mut region = region_with_shape([1, 3, 1]);
        region
            .set_block([0, 0, 0], &parse_block("minecraft:soul_sand"))
            .unwrap();
        region
            .set_block(
                [0, 1, 0],
                &parse_block("minecraft:bubble_column[drag=false]"),
            )
            .unwrap();
        let world = LoadedSchematic {
            name: "bubble-column-surface".to_string(),
            region,
            approximate_collision_blocks: Vec::new(),
        };
        let command = VerifyCommand {
            input: std::path::PathBuf::from("bubble-column-surface.litematic"),
            out: std::path::PathBuf::from("artifacts/test"),
            target_speed: 0.0,
            ticks: 1,
            inspect_tick: Some(1),
            start_x: 0.5,
            start_y: 1.0,
            start_z: 0.5,
            start_vx: 0.0,
            start_vy: 0.0,
            start_vz: 0.0,
            start_on_ground: false,
            width: VERIFY_DEFAULT_WIDTH,
            height: VERIFY_DEFAULT_HEIGHT,
            entity_id_mod4: 0,
            initial_tick_count: 0,
            entity_rng_seed: None,
            entity_uuid: None,
            bootstrap_fluids: false,
            entity_kind: VerifyEntityKind::Item,
            no_ai: false,

            no_gravity: false,
            fire_immune: false,
            start_fire_ticks: 0,
            item_health: None,
        };

        let rows = simulate(&world, &command);
        let tick = &rows[1];
        let expected_vy =
            (BUOYANCY + BUBBLE_COLUMN_SURFACE_ACCELERATION) * VERTICAL_MOVEMENT_DAMPING;
        assert!((tick.x - 0.5).abs() < 1.0e-12);
        assert!((tick.y - (1.0 + BUOYANCY)).abs() < 1.0e-12);
        assert!((tick.vy - expected_vy).abs() < 1.0e-12);
        assert!(tick.in_water);
        assert!(!tick.on_ground);
    }

    #[test]
    fn simulate_applies_bubble_column_internal_boost_when_column_continues_above() {
        let mut region = region_with_shape([1, 4, 1]);
        region
            .set_block([0, 0, 0], &parse_block("minecraft:soul_sand"))
            .unwrap();
        region
            .set_block(
                [0, 1, 0],
                &parse_block("minecraft:bubble_column[drag=false]"),
            )
            .unwrap();
        region
            .set_block(
                [0, 2, 0],
                &parse_block("minecraft:bubble_column[drag=false]"),
            )
            .unwrap();
        let world = LoadedSchematic {
            name: "bubble-column-internal".to_string(),
            region,
            approximate_collision_blocks: Vec::new(),
        };
        let command = VerifyCommand {
            input: std::path::PathBuf::from("bubble-column-internal.litematic"),
            out: std::path::PathBuf::from("artifacts/test"),
            target_speed: 0.0,
            ticks: 1,
            inspect_tick: Some(1),
            start_x: 0.5,
            start_y: 1.0,
            start_z: 0.5,
            start_vx: 0.0,
            start_vy: 0.0,
            start_vz: 0.0,
            start_on_ground: false,
            width: VERIFY_DEFAULT_WIDTH,
            height: VERIFY_DEFAULT_HEIGHT,
            entity_id_mod4: 0,
            initial_tick_count: 0,
            entity_rng_seed: None,
            entity_uuid: None,
            bootstrap_fluids: false,
            entity_kind: VerifyEntityKind::Item,
            no_ai: false,

            no_gravity: false,
            fire_immune: false,
            start_fire_ticks: 0,
            item_health: None,
        };

        let rows = simulate(&world, &command);
        let tick = &rows[1];
        let expected_vy =
            (BUOYANCY + BUBBLE_COLUMN_INTERNAL_ACCELERATION) * VERTICAL_MOVEMENT_DAMPING;
        assert!((tick.y - (1.0 + BUOYANCY)).abs() < 1.0e-12);
        assert!((tick.vy - expected_vy).abs() < 1.0e-12);
        assert!(tick.in_water);
        assert!(!tick.on_ground);
    }

    #[test]
    fn simulate_item_bubble_column_surface_matches_vanilla_probe() {
        let mut region = region_with_shape([1, 3, 1]);
        region
            .set_block([0, 0, 0], &parse_block("minecraft:soul_sand"))
            .unwrap();
        region
            .set_block(
                [0, 1, 0],
                &parse_block("minecraft:bubble_column[drag=false]"),
            )
            .unwrap();
        let world = LoadedSchematic {
            name: "bubble-column-surface-vanilla".to_string(),
            region,
            approximate_collision_blocks: Vec::new(),
        };
        let command = VerifyCommand {
            input: std::path::PathBuf::from("bubble-column-surface-vanilla.litematic"),
            out: std::path::PathBuf::from("artifacts/test"),
            target_speed: 0.0,
            ticks: 1,
            inspect_tick: Some(1),
            start_x: 0.5,
            start_y: 1.0,
            start_z: 0.5,
            start_vx: 0.0,
            start_vy: 0.0,
            start_vz: 0.0,
            start_on_ground: false,
            width: VERIFY_DEFAULT_WIDTH,
            height: VERIFY_DEFAULT_HEIGHT,
            entity_id_mod4: 0,
            initial_tick_count: 0,
            entity_rng_seed: None,
            entity_uuid: None,
            bootstrap_fluids: false,
            entity_kind: VerifyEntityKind::Item,
            no_ai: false,
            no_gravity: false,
            fire_immune: false,
            start_fire_ticks: 0,
            item_health: None,
        };

        let rows = simulate(&world, &command);
        let tick = &rows[1];
        assert!((tick.x - 0.5).abs() < 5.0e-8);
        assert!((tick.y - 1.000_500_000_023_75).abs() < 5.0e-8);
        assert!((tick.z - 0.5).abs() < 5.0e-8);
        assert!((tick.vx - 0.0).abs() < 5.0e-8);
        assert!((tick.vy - 0.098_490_001_940_159_14).abs() < 5.0e-8);
        assert!((tick.vz - 0.0).abs() < 5.0e-8);
        assert!(!tick.on_ground);
        assert!(tick.alive);
    }

    #[test]
    fn simulate_item_bubble_column_internal_matches_vanilla_probe() {
        let mut region = region_with_shape([1, 4, 1]);
        region
            .set_block([0, 0, 0], &parse_block("minecraft:soul_sand"))
            .unwrap();
        region
            .set_block(
                [0, 1, 0],
                &parse_block("minecraft:bubble_column[drag=false]"),
            )
            .unwrap();
        region
            .set_block(
                [0, 2, 0],
                &parse_block("minecraft:bubble_column[drag=false]"),
            )
            .unwrap();
        let world = LoadedSchematic {
            name: "bubble-column-internal-vanilla".to_string(),
            region,
            approximate_collision_blocks: Vec::new(),
        };
        let command = VerifyCommand {
            input: std::path::PathBuf::from("bubble-column-internal-vanilla.litematic"),
            out: std::path::PathBuf::from("artifacts/test"),
            target_speed: 0.0,
            ticks: 1,
            inspect_tick: Some(1),
            start_x: 0.5,
            start_y: 1.0,
            start_z: 0.5,
            start_vx: 0.0,
            start_vy: 0.0,
            start_vz: 0.0,
            start_on_ground: false,
            width: VERIFY_DEFAULT_WIDTH,
            height: VERIFY_DEFAULT_HEIGHT,
            entity_id_mod4: 0,
            initial_tick_count: 0,
            entity_rng_seed: None,
            entity_uuid: None,
            bootstrap_fluids: false,
            entity_kind: VerifyEntityKind::Item,
            no_ai: false,
            no_gravity: false,
            fire_immune: false,
            start_fire_ticks: 0,
            item_health: None,
        };

        let rows = simulate(&world, &command);
        let tick = &rows[1];
        assert!((tick.x - 0.5).abs() < 5.0e-8);
        assert!((tick.y - 1.000_500_000_023_75).abs() < 5.0e-8);
        assert!((tick.z - 0.5).abs() < 5.0e-8);
        assert!((tick.vx - 0.0).abs() < 5.0e-8);
        assert!((tick.vy - 0.059_290_001_177_219_67).abs() < 5.0e-8);
        assert!((tick.vz - 0.0).abs() < 5.0e-8);
        assert!(!tick.on_ground);
        assert!(tick.alive);
    }

    #[test]
    fn cobweb_stuck_multiplier_applies_on_next_tick_like_vanilla() {
        let mut region = region_with_shape([4, 3, 1]);
        region
            .set_block([1, 1, 0], &parse_block("minecraft:cobweb"))
            .unwrap();
        let world = LoadedSchematic {
            name: "cobweb-stuck".to_string(),
            region,
            approximate_collision_blocks: Vec::new(),
        };
        let command = VerifyCommand {
            input: std::path::PathBuf::from("cobweb-stuck.litematic"),
            out: std::path::PathBuf::from("artifacts/test"),
            target_speed: 0.0,
            ticks: 2,
            inspect_tick: Some(2),
            start_x: 1.5,
            start_y: 1.2,
            start_z: 0.5,
            start_vx: 0.2,
            start_vy: 0.0,
            start_vz: 0.0,
            start_on_ground: false,
            width: VERIFY_DEFAULT_WIDTH,
            height: VERIFY_DEFAULT_HEIGHT,
            entity_id_mod4: 0,
            initial_tick_count: 0,
            entity_rng_seed: None,
            entity_uuid: None,
            bootstrap_fluids: false,
            entity_kind: VerifyEntityKind::Item,
            no_ai: false,

            no_gravity: false,
            fire_immune: false,
            start_fire_ticks: 0,
            item_health: None,
        };

        let rows = simulate(&world, &command);
        let tick1 = &rows[1];
        let tick2 = &rows[2];
        assert!((tick1.x - 1.7).abs() < 1.0e-12);
        assert!((tick1.y - 1.16).abs() < 1.0e-12);
        assert!((tick1.vx - (0.2 * HORIZONTAL_MOVEMENT_DAMPING)).abs() < 1.0e-12);
        assert!((tick1.vy - (-0.04 * VERTICAL_MOVEMENT_DAMPING)).abs() < 1.0e-12);

        let expected_tick2_x = tick1.x + tick1.vx * 0.25;
        let expected_tick2_y = tick1.y + (tick1.vy - GRAVITY) * (0.05_f32 as f64);
        assert!((tick2.x - expected_tick2_x).abs() < 1.0e-12);
        assert!((tick2.y - expected_tick2_y).abs() < 1.0e-12);
        assert!((tick2.vx - 0.0).abs() < 1.0e-12);
        assert!((tick2.vy - 0.0).abs() < 1.0e-12);
        assert!(!tick2.on_ground);
    }

    #[test]
    fn simulate_item_cobweb_two_ticks_matches_vanilla_probe() {
        let mut region = region_with_shape([4, 3, 1]);
        region
            .set_block([1, 1, 0], &parse_block("minecraft:cobweb"))
            .unwrap();
        let world = LoadedSchematic {
            name: "cobweb-stuck-vanilla".to_string(),
            region,
            approximate_collision_blocks: Vec::new(),
        };
        let command = VerifyCommand {
            input: std::path::PathBuf::from("cobweb-stuck-vanilla.litematic"),
            out: std::path::PathBuf::from("artifacts/test"),
            target_speed: 0.0,
            ticks: 2,
            inspect_tick: Some(2),
            start_x: 1.5,
            start_y: 1.2,
            start_z: 0.5,
            start_vx: 0.2,
            start_vy: 0.0,
            start_vz: 0.0,
            start_on_ground: false,
            width: VERIFY_DEFAULT_WIDTH,
            height: VERIFY_DEFAULT_HEIGHT,
            entity_id_mod4: 0,
            initial_tick_count: 0,
            entity_rng_seed: None,
            entity_uuid: None,
            bootstrap_fluids: false,
            entity_kind: VerifyEntityKind::Item,
            no_ai: false,
            no_gravity: false,
            fire_immune: false,
            start_fire_ticks: 0,
            item_health: None,
        };

        let rows = simulate(&world, &command);
        let tick1 = &rows[1];
        let tick2 = &rows[2];
        assert!((tick1.x - 1.7).abs() < 5.0e-8);
        assert!((tick1.y - 1.16).abs() < 5.0e-8);
        assert!((tick1.vx - 0.196_000_003_814_697_3).abs() < 5.0e-8);
        assert!((tick1.vy - (-0.039_200_000_762_939_45)).abs() < 5.0e-8);
        assert!((tick2.x - 1.749_000_000_953_674_4).abs() < 5.0e-8);
        assert!((tick2.y - 1.156_039_999_902_844).abs() < 5.0e-8);
        assert!((tick2.vx - 0.0).abs() < 5.0e-8);
        assert!((tick2.vy - 0.0).abs() < 5.0e-8);
        assert!(!tick2.on_ground);
        assert!(tick2.alive);
    }

    #[test]
    fn simulate_item_powder_snow_two_ticks_matches_vanilla_probe() {
        let mut region = region_with_shape([4, 3, 1]);
        region
            .set_block([1, 1, 0], &parse_block("minecraft:powder_snow"))
            .unwrap();
        let world = LoadedSchematic {
            name: "powder-snow-stuck-vanilla".to_string(),
            region,
            approximate_collision_blocks: Vec::new(),
        };
        let command = VerifyCommand {
            input: std::path::PathBuf::from("powder-snow-stuck-vanilla.litematic"),
            out: std::path::PathBuf::from("artifacts/test"),
            target_speed: 0.0,
            ticks: 2,
            inspect_tick: Some(2),
            start_x: 1.5,
            start_y: 1.2,
            start_z: 0.5,
            start_vx: 0.2,
            start_vy: 0.0,
            start_vz: 0.0,
            start_on_ground: false,
            width: VERIFY_DEFAULT_WIDTH,
            height: VERIFY_DEFAULT_HEIGHT,
            entity_id_mod4: 0,
            initial_tick_count: 0,
            entity_rng_seed: None,
            entity_uuid: None,
            bootstrap_fluids: false,
            entity_kind: VerifyEntityKind::Item,
            no_ai: false,
            no_gravity: false,
            fire_immune: false,
            start_fire_ticks: 0,
            item_health: None,
        };

        let rows = simulate(&world, &command);
        let tick1 = &rows[1];
        let tick2 = &rows[2];
        assert!((tick1.x - 1.7).abs() < 5.0e-8);
        assert!((tick1.y - 1.16).abs() < 5.0e-8);
        assert!((tick1.vx - 0.196_000_003_814_697_3).abs() < 5.0e-8);
        assert!((tick1.vy - (-0.039_200_000_762_939_45)).abs() < 5.0e-8);
        assert!((tick2.x - 1.876_399_998_760_223_3).abs() < 5.0e-8);
        assert!((tick2.y - 1.041_199_998_855_59).abs() < 5.0e-8);
        assert!((tick2.vx - 0.0).abs() < 5.0e-8);
        assert!((tick2.vy - 0.0).abs() < 5.0e-8);
        assert!(!tick2.on_ground);
        assert!(tick2.alive);
    }

    #[test]
    fn powder_snow_stuck_multiplier_applies_on_next_tick_like_vanilla() {
        let mut region = region_with_shape([4, 3, 1]);
        region
            .set_block([1, 1, 0], &parse_block("minecraft:powder_snow"))
            .unwrap();
        let world = LoadedSchematic {
            name: "powder-snow-stuck".to_string(),
            region,
            approximate_collision_blocks: Vec::new(),
        };
        let command = VerifyCommand {
            input: std::path::PathBuf::from("powder-snow-stuck.litematic"),
            out: std::path::PathBuf::from("artifacts/test"),
            target_speed: 0.0,
            ticks: 2,
            inspect_tick: Some(2),
            start_x: 1.5,
            start_y: 1.2,
            start_z: 0.5,
            start_vx: 0.2,
            start_vy: 0.0,
            start_vz: 0.0,
            start_on_ground: false,
            width: VERIFY_DEFAULT_WIDTH,
            height: VERIFY_DEFAULT_HEIGHT,
            entity_id_mod4: 0,
            initial_tick_count: 0,
            entity_rng_seed: None,
            entity_uuid: None,
            bootstrap_fluids: false,
            entity_kind: VerifyEntityKind::Item,
            no_ai: false,

            no_gravity: false,
            fire_immune: false,
            start_fire_ticks: 0,
            item_health: None,
        };

        let rows = simulate(&world, &command);
        let tick1 = &rows[1];
        let tick2 = &rows[2];
        let expected_tick2_x = tick1.x + tick1.vx * (0.9_f32 as f64);
        let expected_tick2_y = tick1.y + (tick1.vy - GRAVITY) * 1.5;
        assert!((tick2.x - expected_tick2_x).abs() < 1.0e-12);
        assert!((tick2.y - expected_tick2_y).abs() < 1.0e-12);
        assert!((tick2.vx - 0.0).abs() < 1.0e-12);
        assert!((tick2.vy - 0.0).abs() < 1.0e-12);
        assert!(!tick2.on_ground);
    }

    #[test]
    fn make_stuck_in_block_resets_fall_distance_like_vanilla() {
        let mut region = region_with_shape([1, 1, 1]);
        region
            .set_block([0, 0, 0], &parse_block("minecraft:powder_snow"))
            .unwrap();
        let mut velocity = Vec3d::new(0.0, -0.2, 0.0);
        let mut stuck_speed_multiplier = Vec3d::ZERO;
        let mut fall_distance = 7.0_f64;
        let mut remaining_fire_ticks = 0;
        let mut item_health = default_item_health(VerifyEntityKind::Item);

        let removed = apply_inside_block_effects(
            &region,
            Vec3d::new(0.5, 0.2, 0.5),
            Vec3d::new(0.5, 0.2, 0.5),
            VERIFY_DEFAULT_WIDTH,
            VERIFY_DEFAULT_HEIGHT,
            VerifyEntityKind::Item,
            false,
            false,
            &mut velocity,
            &mut stuck_speed_multiplier,
            &mut fall_distance,
            &mut remaining_fire_ticks,
            &mut item_health,
        );

        assert!(removed.is_none());
        assert_eq!(fall_distance, 0.0);
        assert!((stuck_speed_multiplier.x - 0.9_f32 as f64).abs() < 1.0e-12);
        assert!((stuck_speed_multiplier.y - 1.5).abs() < 1.0e-12);
        assert!((stuck_speed_multiplier.z - 0.9_f32 as f64).abs() < 1.0e-12);
    }

    #[test]
    fn long_move_through_water_resets_fall_distance() {
        let mut region = region_with_shape([1, 3, 1]);
        region
            .set_block([0, 0, 0], &parse_block("minecraft:water"))
            .unwrap();
        assert!(movement_resets_fall_distance(
            &region,
            Vec3d::new(0.5, 2.5, 0.5),
            Vec3d::new(0.0, -3.0, 0.0),
        ));
    }

    #[test]
    fn long_move_through_cobweb_resets_fall_distance() {
        let mut region = region_with_shape([1, 3, 1]);
        region
            .set_block([0, 1, 0], &parse_block("minecraft:cobweb"))
            .unwrap();
        assert!(movement_resets_fall_distance(
            &region,
            Vec3d::new(0.5, 2.5, 0.5),
            Vec3d::new(0.0, -2.0, 0.0),
        ));
    }

    #[test]
    fn climbable_blocks_reset_fall_distance() {
        let mut region = region_with_shape([1, 3, 1]);
        region
            .set_block([0, 1, 0], &parse_block("minecraft:ladder[facing=north]"))
            .unwrap();
        assert!(movement_resets_fall_distance(
            &region,
            Vec3d::new(0.5, 2.5, 0.5),
            Vec3d::new(0.0, -2.0, 0.0),
        ));
    }

    #[test]
    fn scaffolding_resets_fall_distance_during_long_move() {
        let mut region = region_with_shape([1, 3, 1]);
        region
            .set_block(
                [0, 1, 0],
                &parse_block("minecraft:scaffolding[bottom=false,distance=0,waterlogged=false]"),
            )
            .unwrap();
        assert!(movement_resets_fall_distance(
            &region,
            Vec3d::new(0.5, 2.5, 0.5),
            Vec3d::new(0.0, -2.0, 0.0),
        ));
    }

    #[test]
    fn fall_distance_reset_clip_caps_distance_at_eight_blocks() {
        let mut region = region_with_shape([1, 12, 1]);
        region
            .set_block([0, 1, 0], &parse_block("minecraft:water"))
            .unwrap();
        assert!(!movement_resets_fall_distance(
            &region,
            Vec3d::new(0.5, 10.5, 0.5),
            Vec3d::new(0.0, -20.0, 0.0),
        ));
    }

    #[test]
    fn powder_snow_collision_requires_fall_distance_above_threshold() {
        let mut region = region_with_shape([1, 1, 1]);
        region
            .set_block([0, 0, 0], &parse_block("minecraft:powder_snow"))
            .unwrap();

        let pass_through = move_entity_with_fall_distance(
            &region,
            Vec3d::new(0.5, 1.0, 0.5),
            Vec3d::new(0.0, -1.0, 0.0),
            VERIFY_DEFAULT_WIDTH,
            VERIFY_DEFAULT_HEIGHT,
            POWDER_SNOW_FALL_DISTANCE_COLLISION_THRESHOLD,
            VerifyEntityKind::Item,
        );
        assert!(!pass_through.collided_y);
        assert!((pass_through.delta.y + 1.0).abs() < 1.0e-12);

        let collide = move_entity_with_fall_distance(
            &region,
            Vec3d::new(0.5, 1.0, 0.5),
            Vec3d::new(0.0, -1.0, 0.0),
            VERIFY_DEFAULT_WIDTH,
            VERIFY_DEFAULT_HEIGHT,
            POWDER_SNOW_FALL_DISTANCE_COLLISION_THRESHOLD + 1.0e-6,
            VerifyEntityKind::Item,
        );
        assert!(collide.collided_y);
        assert!(
            (collide.delta.y - ((0.9_f32 as f64) - 1.0)).abs() < 1.0e-12,
            "unexpected collide delta: {}",
            collide.delta.y
        );
    }

    #[test]
    fn falling_blocks_walk_on_powder_snow_without_fall_distance_threshold() {
        let mut region = region_with_shape([1, 1, 1]);
        region
            .set_block([0, 0, 0], &parse_block("minecraft:powder_snow"))
            .unwrap();

        let collide = move_entity_with_fall_distance(
            &region,
            Vec3d::new(0.5, 1.0, 0.5),
            Vec3d::new(0.0, -1.0, 0.0),
            VERIFY_DEFAULT_WIDTH,
            VERIFY_DEFAULT_HEIGHT,
            0.0,
            VerifyEntityKind::FallingBlock,
        );
        assert!(collide.collided_y);
        assert!(collide.delta.y.abs() < 1.0e-12);
    }

    #[test]
    fn living_entities_get_powder_snow_stuck_when_feet_block_is_powder_snow() {
        let mut region = region_with_shape([1, 2, 1]);
        region
            .set_block([0, 0, 0], &parse_block("minecraft:powder_snow"))
            .unwrap();
        let mut velocity = Vec3d::new(0.0, -0.2, 0.0);
        let mut stuck_speed_multiplier = Vec3d::ZERO;
        let mut fall_distance = 7.0_f64;
        let mut remaining_fire_ticks = 0;
        let mut item_health = None;

        let removed = apply_inside_block_effects(
            &region,
            Vec3d::new(0.5, 0.9, 0.5),
            Vec3d::new(0.5, 0.9, 0.5),
            VERIFY_DEFAULT_WIDTH,
            VERIFY_DEFAULT_HEIGHT,
            VerifyEntityKind::Living,
            false,
            false,
            &mut velocity,
            &mut stuck_speed_multiplier,
            &mut fall_distance,
            &mut remaining_fire_ticks,
            &mut item_health,
        );

        assert!(removed.is_none());
        assert!((stuck_speed_multiplier.x - 0.9_f32 as f64).abs() < 1.0e-12);
        assert!((stuck_speed_multiplier.y - 1.5).abs() < 1.0e-12);
        assert!((stuck_speed_multiplier.z - 0.9_f32 as f64).abs() < 1.0e-12);
        assert_eq!(fall_distance, 0.0);
    }

    #[test]
    fn living_entities_do_not_get_powder_snow_stuck_when_feet_are_above_block() {
        let mut region = region_with_shape([1, 1, 1]);
        region
            .set_block([0, 0, 0], &parse_block("minecraft:powder_snow"))
            .unwrap();
        let mut velocity = Vec3d::new(0.0, -0.2, 0.0);
        let mut stuck_speed_multiplier = Vec3d::ZERO;
        let mut fall_distance = 7.0_f64;
        let mut remaining_fire_ticks = 0;
        let mut item_health = None;

        let removed = apply_inside_block_effects(
            &region,
            Vec3d::new(0.5, 1.0, 0.5),
            Vec3d::new(0.5, 1.0, 0.5),
            VERIFY_DEFAULT_WIDTH,
            VERIFY_DEFAULT_HEIGHT,
            VerifyEntityKind::Living,
            false,
            false,
            &mut velocity,
            &mut stuck_speed_multiplier,
            &mut fall_distance,
            &mut remaining_fire_ticks,
            &mut item_health,
        );

        assert!(removed.is_none());
        assert!(stuck_speed_multiplier.length_sqr() < 1.0e-18);
        assert!((fall_distance - 7.0).abs() < 1.0e-12);
    }

    #[test]
    fn sweet_berry_bush_is_non_solid_for_collision() {
        let mut region = region_with_shape([3, 2, 1]);
        region
            .set_block([1, 0, 0], &parse_block("minecraft:sweet_berry_bush[age=2]"))
            .unwrap();
        let move_result = move_entity(
            &region,
            Vec3d::new(0.5, 0.0, 0.5),
            Vec3d::new(1.0, 0.0, 0.0),
            VERIFY_DEFAULT_WIDTH,
            VERIFY_DEFAULT_HEIGHT,
        );
        assert!(!move_result.collided_x);
        assert!((move_result.delta.x - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn scaffolding_supports_entities_from_above() {
        let mut region = region_with_shape([1, 1, 1]);
        region
            .set_block(
                [0, 0, 0],
                &parse_block("minecraft:scaffolding[bottom=false,distance=0,waterlogged=false]"),
            )
            .unwrap();
        let move_result = move_entity(
            &region,
            Vec3d::new(0.5, 1.5, 0.5),
            Vec3d::new(0.0, -1.0, 0.0),
            VERIFY_DEFAULT_WIDTH,
            VERIFY_DEFAULT_HEIGHT,
        );
        assert!(move_result.collided_y);
        assert!((move_result.delta.y + 0.5).abs() < 1.0e-12);
    }

    #[test]
    fn scaffolding_is_non_solid_when_entered_from_inside() {
        let mut region = region_with_shape([3, 1, 1]);
        region
            .set_block(
                [1, 0, 0],
                &parse_block("minecraft:scaffolding[bottom=false,distance=0,waterlogged=false]"),
            )
            .unwrap();
        let move_result = move_entity(
            &region,
            Vec3d::new(0.5, 0.5, 0.5),
            Vec3d::new(1.0, 0.0, 0.0),
            VERIFY_DEFAULT_WIDTH,
            VERIFY_DEFAULT_HEIGHT,
        );
        assert!(!move_result.collided_x);
        assert!((move_result.delta.x - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn scaffolding_bottom_plate_collides_when_bottom_flag_is_set() {
        let mut region = region_with_shape([1, 1, 1]);
        region
            .set_block(
                [0, 0, 0],
                &parse_block("minecraft:scaffolding[bottom=true,distance=1,waterlogged=false]"),
            )
            .unwrap();
        let move_result = move_entity(
            &region,
            Vec3d::new(0.5, 0.3, 0.5),
            Vec3d::new(0.0, -0.3, 0.0),
            VERIFY_DEFAULT_WIDTH,
            VERIFY_DEFAULT_HEIGHT,
        );
        assert!(move_result.collided_y);
        assert!((move_result.delta.y + 0.175).abs() < 1.0e-12);
    }

    #[test]
    fn campfire_supports_entities_at_seven_sixteenths_height() {
        let mut region = region_with_shape([1, 1, 1]);
        region
            .set_block(
                [0, 0, 0],
                &parse_block(
                    "minecraft:campfire[facing=north,lit=true,signal_fire=false,waterlogged=false]",
                ),
            )
            .unwrap();
        let move_result = move_entity(
            &region,
            Vec3d::new(0.5, 1.5, 0.5),
            Vec3d::new(0.0, -2.0, 0.0),
            VERIFY_DEFAULT_WIDTH,
            VERIFY_DEFAULT_HEIGHT,
        );
        assert!(move_result.collided_y);
        assert!((move_result.delta.y + (17.0 / 16.0)).abs() < 1.0e-12);
    }

    #[test]
    fn mature_sweet_berry_bush_sticks_living_entities_only() {
        let mut region = region_with_shape([1, 1, 1]);
        region
            .set_block([0, 0, 0], &parse_block("minecraft:sweet_berry_bush[age=2]"))
            .unwrap();
        let mut velocity = Vec3d::new(0.1, 0.0, 0.0);
        let mut stuck_speed_multiplier = Vec3d::ZERO;
        let mut fall_distance = 3.0_f64;
        let mut remaining_fire_ticks = 0;
        let mut item_health = None;

        let removed = apply_inside_block_effects(
            &region,
            Vec3d::new(0.5, 0.2, 0.5),
            Vec3d::new(0.5, 0.2, 0.5),
            VERIFY_DEFAULT_WIDTH,
            VERIFY_DEFAULT_HEIGHT,
            VerifyEntityKind::Living,
            false,
            false,
            &mut velocity,
            &mut stuck_speed_multiplier,
            &mut fall_distance,
            &mut remaining_fire_ticks,
            &mut item_health,
        );

        assert!(removed.is_none());
        assert!((stuck_speed_multiplier.x - 0.8_f32 as f64).abs() < 1.0e-12);
        assert!((stuck_speed_multiplier.y - 0.75).abs() < 1.0e-12);
        assert!((stuck_speed_multiplier.z - 0.8_f32 as f64).abs() < 1.0e-12);
        assert_eq!(fall_distance, 0.0);

        let mut non_living_multiplier = Vec3d::ZERO;
        let mut non_living_fall_distance = 3.0_f64;
        let mut non_living_fire_ticks = 0;
        let mut non_living_item_health = None;
        let _ = apply_inside_block_effects(
            &region,
            Vec3d::new(0.5, 0.2, 0.5),
            Vec3d::new(0.5, 0.2, 0.5),
            VERIFY_DEFAULT_WIDTH,
            VERIFY_DEFAULT_HEIGHT,
            VerifyEntityKind::Generic,
            false,
            false,
            &mut velocity,
            &mut non_living_multiplier,
            &mut non_living_fall_distance,
            &mut non_living_fire_ticks,
            &mut non_living_item_health,
        );
        assert!(non_living_multiplier.length_sqr() < 1.0e-18);
        assert!((non_living_fall_distance - 3.0).abs() < 1.0e-12);
    }

    #[test]
    fn generic_entities_do_not_use_item_motion_sampling_gate() {
        let mut region = region_with_shape([2, 1, 1]);
        region
            .set_block([0, 0, 0], &parse_block("minecraft:stone"))
            .unwrap();
        region
            .set_block([1, 0, 0], &parse_block("minecraft:stone"))
            .unwrap();
        let world = LoadedSchematic {
            name: "generic-motion-gate".to_string(),
            region,
            approximate_collision_blocks: Vec::new(),
        };

        let item_rows = simulate(
            &world,
            &VerifyCommand {
                input: std::path::PathBuf::from("generic-motion-gate.litematic"),
                out: std::path::PathBuf::from("artifacts/test"),
                target_speed: 0.0,
                ticks: 1,
                inspect_tick: Some(1),
                start_x: 0.5,
                start_y: 1.0,
                start_z: 0.5,
                start_vx: 0.001,
                start_vy: 0.0,
                start_vz: 0.0,
                start_on_ground: true,
                width: VERIFY_DEFAULT_WIDTH,
                height: VERIFY_DEFAULT_HEIGHT,
                entity_id_mod4: 0,
                initial_tick_count: 0,
                entity_rng_seed: None,
                entity_uuid: None,
                bootstrap_fluids: false,
                entity_kind: VerifyEntityKind::Item,
                no_ai: false,

                no_gravity: false,
                fire_immune: false,
                start_fire_ticks: 0,
                item_health: None,
            },
        );
        let generic_rows = simulate(
            &world,
            &VerifyCommand {
                input: std::path::PathBuf::from("generic-motion-gate.litematic"),
                out: std::path::PathBuf::from("artifacts/test"),
                target_speed: 0.0,
                ticks: 1,
                inspect_tick: Some(1),
                start_x: 0.5,
                start_y: 1.0,
                start_z: 0.5,
                start_vx: 0.001,
                start_vy: 0.0,
                start_vz: 0.0,
                start_on_ground: true,
                width: VERIFY_DEFAULT_WIDTH,
                height: VERIFY_DEFAULT_HEIGHT,
                entity_id_mod4: 0,
                initial_tick_count: 0,
                entity_rng_seed: None,
                entity_uuid: None,
                bootstrap_fluids: false,
                entity_kind: VerifyEntityKind::Generic,
                no_ai: false,

                no_gravity: false,
                fire_immune: false,
                start_fire_ticks: 0,
                item_health: None,
            },
        );

        assert!((item_rows[1].x - 0.5).abs() < 1.0e-12);
        assert!(generic_rows[1].x > 0.5 + 1.0e-6);
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
        let mut block_ticks = DynamicBlockTicks::default();
        fluid_ticks.spread_to(
            &mut region,
            &mut block_ticks,
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
    fn spread_to_wall_sign_keeps_block_and_sets_waterlogged() {
        let mut region = region_with_shape([1, 1, 1]);
        let sign = parse_block("minecraft:oak_wall_sign[facing=north,waterlogged=false]");
        region.set_block([0, 0, 0], &sign).unwrap();
        let mut fluid_ticks = DynamicFluidTicks::default();
        let mut block_ticks = DynamicBlockTicks::default();
        fluid_ticks.spread_to(
            &mut region,
            &mut block_ticks,
            [0, 0, 0],
            &sign,
            DynamicWaterState {
                amount: 8,
                falling: false,
            },
            0,
        );
        let block = block_at(&region, [0, 0, 0]).expect("waterlogged wall sign");
        assert_eq!(block.id, "oak_wall_sign");
        assert_eq!(
            block.attributes.get("waterlogged").map(String::as_str),
            Some("true")
        );
    }

    #[test]
    fn flowing_water_cannot_be_placed_in_sign() {
        let sign = parse_block("minecraft:oak_wall_sign[facing=north,waterlogged=false]");
        assert!(!can_hold_specific_fluid(
            &sign,
            DynamicWaterState {
                amount: 7,
                falling: false,
            }
        ));
    }

    #[test]
    fn waterlogged_sign_stays_waterlogged_after_source_tick() {
        let mut region = region_with_shape([1, 1, 1]);
        let sign = parse_block("minecraft:oak_wall_sign[facing=north,waterlogged=true]");
        region.set_block([0, 0, 0], &sign).unwrap();
        let mut fluid_ticks = DynamicFluidTicks::default();
        let mut block_ticks = DynamicBlockTicks::default();
        fluid_ticks.tick_water(&mut region, &mut block_ticks, [0, 0, 0], 0);
        let block = block_at(&region, [0, 0, 0]).expect("waterlogged sign remains present");
        assert_eq!(block.id, "oak_wall_sign");
        assert_eq!(
            block.attributes.get("waterlogged").map(String::as_str),
            Some("true")
        );
    }

    #[test]
    fn signs_do_not_block_entity_motion() {
        let mut region = region_with_shape([3, 2, 1]);
        region
            .set_block(
                [1, 0, 0],
                &parse_block("minecraft:oak_wall_sign[facing=north,waterlogged=false]"),
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
    fn non_collision_blocks_do_not_block_entity_motion() {
        for block_id in [
            "minecraft:oak_sign[rotation=0,waterlogged=false]",
            "minecraft:oak_wall_hanging_sign[facing=north,waterlogged=false]",
            "minecraft:oak_hanging_sign[rotation=0,attached=false,waterlogged=false]",
            "minecraft:oak_hanging_sign[rotation=2,attached=false,waterlogged=false]",
            "minecraft:oak_pressure_plate[powered=false]",
            "minecraft:white_banner[rotation=0]",
            "minecraft:white_wall_banner[facing=north]",
        ] {
            let mut region = region_with_shape([3, 2, 1]);
            region.set_block([1, 0, 0], &parse_block(block_id)).unwrap();
            let move_result = move_entity(
                &region,
                Vec3d::new(0.5, 0.0, 0.5),
                Vec3d::new(1.0, 0.0, 0.0),
                0.25,
                0.25,
            );
            assert!(
                !move_result.collided_x,
                "unexpected collision for {block_id}"
            );
            assert!(
                (move_result.delta.x - 1.0).abs() < 1.0e-12,
                "unexpected delta for {block_id}"
            );
        }
    }

    #[test]
    fn hanging_signs_and_ladder_can_be_waterlogged() {
        for block_id in [
            "minecraft:oak_wall_hanging_sign[facing=east,waterlogged=false]",
            "minecraft:oak_hanging_sign[rotation=0,attached=false,waterlogged=false]",
            "minecraft:ladder[facing=north,waterlogged=false]",
            "minecraft:iron_chain[axis=y,waterlogged=false]",
            "minecraft:lightning_rod[facing=up,powered=false,waterlogged=false]",
        ] {
            let mut region = region_with_shape([1, 1, 1]);
            let block = parse_block(block_id);
            region.set_block([0, 0, 0], &block).unwrap();
            let mut fluid_ticks = DynamicFluidTicks::default();
            let mut block_ticks = DynamicBlockTicks::default();
            fluid_ticks.spread_to(
                &mut region,
                &mut block_ticks,
                [0, 0, 0],
                &block,
                DynamicWaterState {
                    amount: 8,
                    falling: false,
                },
                0,
            );
            let updated = block_at(&region, [0, 0, 0]).expect("waterlogged block remains");
            assert_eq!(
                updated.id, block.id,
                "unexpected replacement for {block_id}"
            );
            assert_eq!(
                updated.attributes.get("waterlogged").map(String::as_str),
                Some("true"),
                "expected waterlogged state for {block_id}"
            );
        }
    }

    #[test]
    fn scaffolding_can_be_waterlogged() {
        let mut region = region_with_shape([1, 1, 1]);
        let scaffolding =
            parse_block("minecraft:scaffolding[bottom=false,distance=0,waterlogged=false]");
        region.set_block([0, 0, 0], &scaffolding).unwrap();
        let mut fluid_ticks = DynamicFluidTicks::default();
        let mut block_ticks = DynamicBlockTicks::default();
        fluid_ticks.spread_to(
            &mut region,
            &mut block_ticks,
            [0, 0, 0],
            &scaffolding,
            DynamicWaterState {
                amount: 8,
                falling: false,
            },
            0,
        );
        let updated = block_at(&region, [0, 0, 0]).expect("waterlogged scaffolding remains");
        assert_eq!(updated.id, "scaffolding");
        assert_eq!(
            updated.attributes.get("waterlogged").map(String::as_str),
            Some("true")
        );
    }

    #[test]
    fn decorated_pot_and_chests_can_be_waterlogged() {
        for block_id in [
            "minecraft:decorated_pot[facing=north,cracked=false,waterlogged=false]",
            "minecraft:chest[facing=north,type=single,waterlogged=false]",
            "minecraft:trapped_chest[facing=north,type=single,waterlogged=false]",
        ] {
            let mut region = region_with_shape([1, 1, 1]);
            let block = parse_block(block_id);
            region.set_block([0, 0, 0], &block).unwrap();
            let mut fluid_ticks = DynamicFluidTicks::default();
            let mut block_ticks = DynamicBlockTicks::default();
            fluid_ticks.spread_to(
                &mut region,
                &mut block_ticks,
                [0, 0, 0],
                &block,
                DynamicWaterState {
                    amount: 8,
                    falling: false,
                },
                0,
            );
            let updated = block_at(&region, [0, 0, 0]).expect("waterlogged block remains");
            assert_eq!(
                updated.id, block.id,
                "unexpected replacement for {block_id}"
            );
            assert_eq!(
                updated.attributes.get("waterlogged").map(String::as_str),
                Some("true"),
                "expected waterlogged state for {block_id}"
            );
        }
    }

    #[test]
    fn big_dripleaf_family_can_be_waterlogged() {
        for block_id in [
            "minecraft:big_dripleaf[facing=north,tilt=none,waterlogged=false]",
            "minecraft:big_dripleaf_stem[facing=north,waterlogged=false]",
        ] {
            let mut region = region_with_shape([1, 1, 1]);
            let block = parse_block(block_id);
            region.set_block([0, 0, 0], &block).unwrap();
            let mut fluid_ticks = DynamicFluidTicks::default();
            let mut block_ticks = DynamicBlockTicks::default();
            fluid_ticks.spread_to(
                &mut region,
                &mut block_ticks,
                [0, 0, 0],
                &block,
                DynamicWaterState {
                    amount: 8,
                    falling: false,
                },
                0,
            );
            let updated = block_at(&region, [0, 0, 0]).expect("waterlogged dripleaf remains");
            assert_eq!(
                updated.id, block.id,
                "unexpected replacement for {block_id}"
            );
            assert_eq!(
                updated.attributes.get("waterlogged").map(String::as_str),
                Some("true"),
                "expected waterlogged state for {block_id}"
            );
        }
    }

    #[test]
    fn pointed_dripstone_can_be_waterlogged() {
        let mut region = region_with_shape([1, 1, 1]);
        let block = parse_block(
            "minecraft:pointed_dripstone[thickness=tip,vertical_direction=up,waterlogged=false]",
        );
        region.set_block([0, 0, 0], &block).unwrap();
        let mut fluid_ticks = DynamicFluidTicks::default();
        let mut block_ticks = DynamicBlockTicks::default();
        fluid_ticks.spread_to(
            &mut region,
            &mut block_ticks,
            [0, 0, 0],
            &block,
            DynamicWaterState {
                amount: 8,
                falling: false,
            },
            0,
        );
        let updated = block_at(&region, [0, 0, 0]).expect("waterlogged dripstone remains");
        assert_eq!(updated.id, "pointed_dripstone");
        assert_eq!(
            updated.attributes.get("waterlogged").map(String::as_str),
            Some("true")
        );
    }

    #[test]
    fn unsupported_upward_pointed_dripstone_breaks_on_first_tick() {
        let mut region = region_with_shape([1, 2, 1]);
        region
            .set_block(
                [0, 1, 0],
                &parse_block(
                    "minecraft:pointed_dripstone[thickness=tip,vertical_direction=up,waterlogged=false]",
                ),
            )
            .unwrap();

        run_world_ticks(&mut region, 1);

        assert_eq!(
            block_full_id(
                block_at(&region, [0, 1, 0]).expect("air replaces unsupported dripstone")
            ),
            "minecraft:air"
        );
    }

    #[test]
    fn fluid_flow_looks_through_scaffolding_for_below_neighbor_water() {
        let mut region = region_with_shape([2, 2, 1]);
        region
            .set_block([0, 1, 0], &parse_block("minecraft:water"))
            .unwrap();
        region
            .set_block(
                [1, 1, 0],
                &parse_block("minecraft:scaffolding[bottom=false,distance=0,waterlogged=false]"),
            )
            .unwrap();
        region
            .set_block([1, 0, 0], &parse_block("minecraft:water"))
            .unwrap();

        let scaffolding_flow = fluid_flow(
            &region,
            [0, 1, 0],
            water_at(&region, [0, 1, 0]).expect("water"),
            FluidKind::Water,
        );
        assert!(scaffolding_flow.x > 0.9);

        region
            .set_block([1, 1, 0], &parse_block("minecraft:stone"))
            .unwrap();
        let stone_flow = fluid_flow(
            &region,
            [0, 1, 0],
            water_at(&region, [0, 1, 0]).expect("water"),
            FluidKind::Water,
        );
        assert!(stone_flow.x.abs() < 1.0e-12);
    }

    #[test]
    fn source_conversion_matches_vanilla_floor_support_rules() {
        let mut solid_floor = region_with_shape([3, 3, 1]);
        for x in 0..=2 {
            solid_floor
                .set_block([x, 0, 0], &parse_block("minecraft:stone"))
                .unwrap();
            solid_floor
                .set_block([x, 1, 0], &parse_block("minecraft:stone"))
                .unwrap();
        }
        solid_floor
            .set_block([0, 2, 0], &parse_block("minecraft:water"))
            .unwrap();
        solid_floor
            .set_block([2, 2, 0], &parse_block("minecraft:water"))
            .unwrap();
        run_world_ticks(&mut solid_floor, 5);
        assert!(
            dynamic_water_state_at(&solid_floor, [1, 2, 0])
                .map(|state| state.is_source())
                .unwrap_or(false)
        );
        assert_eq!(
            block_full_id(block_at(&solid_floor, [1, 2, 0]).expect("source water above stone")),
            "minecraft:water"
        );

        let mut slab_floor = region_with_shape([3, 3, 1]);
        for x in 0..=2 {
            slab_floor
                .set_block([x, 0, 0], &parse_block("minecraft:stone"))
                .unwrap();
            if x != 1 {
                slab_floor
                    .set_block([x, 1, 0], &parse_block("minecraft:stone"))
                    .unwrap();
            }
        }
        slab_floor
            .set_block([0, 2, 0], &parse_block("minecraft:water"))
            .unwrap();
        slab_floor
            .set_block([2, 2, 0], &parse_block("minecraft:water"))
            .unwrap();
        slab_floor
            .set_block(
                [1, 1, 0],
                &parse_block("minecraft:oak_slab[type=bottom,waterlogged=false]"),
            )
            .unwrap();
        run_world_ticks(&mut slab_floor, 5);
        assert!(
            dynamic_water_state_at(&slab_floor, [1, 2, 0])
                .map(|state| state.is_source())
                .unwrap_or(false)
        );
        assert_eq!(
            block_full_id(block_at(&slab_floor, [1, 2, 0]).expect("source water above slab")),
            "minecraft:water"
        );
        assert_eq!(
            block_full_id(block_at(&slab_floor, [1, 1, 0]).expect("slab floor remains")),
            "minecraft:oak_slab[type=bottom,waterlogged=false]"
        );

        let mut scaffolding_floor = region_with_shape([3, 3, 1]);
        for x in 0..=2 {
            scaffolding_floor
                .set_block([x, 0, 0], &parse_block("minecraft:stone"))
                .unwrap();
            if x != 1 {
                scaffolding_floor
                    .set_block([x, 1, 0], &parse_block("minecraft:stone"))
                    .unwrap();
            }
        }
        scaffolding_floor
            .set_block([0, 2, 0], &parse_block("minecraft:water"))
            .unwrap();
        scaffolding_floor
            .set_block([2, 2, 0], &parse_block("minecraft:water"))
            .unwrap();
        scaffolding_floor
            .set_block(
                [1, 1, 0],
                &parse_block("minecraft:scaffolding[bottom=false,distance=0,waterlogged=false]"),
            )
            .unwrap();
        run_world_ticks(&mut scaffolding_floor, 5);
        assert!(
            dynamic_water_state_at(&scaffolding_floor, [1, 2, 0]).is_some(),
            "center cell should still fill with water above scaffolding"
        );
        assert!(
            !dynamic_water_state_at(&scaffolding_floor, [1, 2, 0])
                .map(|state| state.is_source())
                .unwrap_or(false)
        );
        assert_eq!(
            block_full_id(block_at(&scaffolding_floor, [1, 2, 0]).expect("flowing center water")),
            "minecraft:water[level=1]"
        );
        assert_eq!(
            block_full_id(
                block_at(&scaffolding_floor, [1, 1, 0]).expect("supported scaffolding remains")
            ),
            "minecraft:scaffolding[bottom=false,distance=0,waterlogged=false]"
        );
    }

    #[test]
    fn unsupported_scaffolding_breaks_on_its_scheduled_tick() {
        let mut region = region_with_shape([1, 2, 1]);
        region
            .set_block(
                [0, 0, 0],
                &parse_block("minecraft:scaffolding[bottom=false,distance=0,waterlogged=false]"),
            )
            .unwrap();
        run_world_ticks(&mut region, 1);
        assert_eq!(
            block_full_id(
                block_at(&region, [0, 0, 0]).expect("air replaces unsupported scaffolding")
            ),
            "minecraft:air"
        );
    }

    #[test]
    fn big_dripleaf_tick_sequence_matches_guardian() {
        let mut region = region_with_shape([1, 2, 1]);
        region
            .set_block([0, 0, 0], &parse_block("minecraft:dirt"))
            .unwrap();
        region
            .set_block(
                [0, 1, 0],
                &parse_block(
                    "minecraft:big_dripleaf[facing=north,tilt=unstable,waterlogged=false]",
                ),
            )
            .unwrap();

        let mut block_ticks = DynamicBlockTicks::bootstrap(&region);
        let mut fluid_ticks = DynamicFluidTicks::default();
        for tick in 1..=120 {
            block_ticks.run_due(&mut region, tick);
            fluid_ticks.run_due(&mut region, &mut block_ticks, tick);
            let block = block_at(&region, [0, 1, 0]).expect("big dripleaf remains");
            let expected_tilt = match tick {
                1..=9 => "unstable",
                10..=19 => "partial",
                20..=119 => "full",
                _ => "none",
            };
            assert_eq!(
                big_dripleaf_tilt(block),
                expected_tilt,
                "unexpected tilt at tick {tick}"
            );
        }
    }

    #[test]
    fn unsupported_big_dripleaf_persists_without_scheduled_tick() {
        let mut region = region_with_shape([1, 2, 1]);
        region
            .set_block(
                [0, 1, 0],
                &parse_block("minecraft:big_dripleaf[facing=north,tilt=none,waterlogged=false]"),
            )
            .unwrap();
        run_world_ticks(&mut region, 3);
        assert_eq!(
            block_full_id(
                block_at(&region, [0, 1, 0])
                    .expect("unsupported dripleaf persists through three snapshot ticks")
            ),
            "minecraft:big_dripleaf[facing=north,tilt=none,waterlogged=false]"
        );
    }

    #[test]
    fn scheduled_unsupported_big_dripleaf_stem_breaks() {
        let mut region = region_with_shape([1, 3, 1]);
        region
            .set_block(
                [0, 1, 0],
                &parse_block("minecraft:big_dripleaf_stem[facing=north,waterlogged=false]"),
            )
            .unwrap();
        let mut block_ticks = DynamicBlockTicks::default();
        block_ticks.schedule(BIG_DRIPLEAF_SURVIVAL_TICK_DELAY, [0, 1, 0]);
        block_ticks.run_due(&mut region, BIG_DRIPLEAF_SURVIVAL_TICK_DELAY);
        assert_eq!(
            block_full_id(block_at(&region, [0, 1, 0]).expect("unsupported dripleaf stem breaks")),
            "minecraft:air"
        );
    }

    #[test]
    fn stacked_big_dripleaf_leaf_converts_lower_leaf_to_stem() {
        let mut region = region_with_shape([1, 3, 1]);
        region
            .set_block([0, 0, 0], &parse_block("minecraft:dirt"))
            .unwrap();
        region
            .set_block(
                [0, 1, 0],
                &parse_block("minecraft:big_dripleaf[facing=north,tilt=none,waterlogged=false]"),
            )
            .unwrap();
        region
            .set_block(
                [0, 2, 0],
                &parse_block("minecraft:big_dripleaf[facing=north,tilt=none,waterlogged=false]"),
            )
            .unwrap();
        run_world_ticks(&mut region, 1);
        assert_eq!(
            block_full_id(block_at(&region, [0, 1, 0]).expect("static snapshot keeps lower leaf")),
            "minecraft:big_dripleaf[facing=north,tilt=none,waterlogged=false]"
        );
        assert_eq!(
            block_full_id(block_at(&region, [0, 2, 0]).expect("upper dripleaf remains leaf")),
            "minecraft:big_dripleaf[facing=north,tilt=none,waterlogged=false]"
        );

        let mut scheduled = region_with_shape([1, 3, 1]);
        scheduled
            .set_block([0, 0, 0], &parse_block("minecraft:dirt"))
            .unwrap();
        scheduled
            .set_block(
                [0, 1, 0],
                &parse_block("minecraft:big_dripleaf[facing=north,tilt=none,waterlogged=false]"),
            )
            .unwrap();
        scheduled
            .set_block(
                [0, 2, 0],
                &parse_block("minecraft:big_dripleaf[facing=north,tilt=none,waterlogged=false]"),
            )
            .unwrap();
        let mut block_ticks = DynamicBlockTicks::default();
        block_ticks.schedule(BIG_DRIPLEAF_SURVIVAL_TICK_DELAY, [0, 1, 0]);
        block_ticks.run_due(&mut scheduled, BIG_DRIPLEAF_SURVIVAL_TICK_DELAY);
        assert_eq!(
            block_full_id(
                block_at(&scheduled, [0, 1, 0]).expect("scheduled lower dripleaf becomes stem")
            ),
            "minecraft:big_dripleaf_stem[facing=north,waterlogged=false]"
        );
    }

    #[test]
    fn powered_big_dripleaf_does_not_tilt_or_stays_tilted() {
        let mut region = region_with_shape([2, 2, 1]);
        region
            .set_block([0, 0, 0], &parse_block("minecraft:dirt"))
            .unwrap();
        region
            .set_block(
                [0, 1, 0],
                &parse_block("minecraft:big_dripleaf[facing=north,tilt=none,waterlogged=false]"),
            )
            .unwrap();
        region
            .set_block([1, 1, 0], &parse_block("minecraft:redstone_block"))
            .unwrap();

        let mut block_ticks = DynamicBlockTicks::default();
        maybe_trigger_big_dripleaf(
            &mut region,
            &mut block_ticks,
            Vec3d::new(0.5, 1.0 + 15.0 / 16.0, 0.5),
            VERIFY_DEFAULT_WIDTH,
            true,
            1,
        );
        let stable = block_at(&region, [0, 1, 0]).expect("powered dripleaf remains");
        assert_eq!(big_dripleaf_tilt(stable), "none");

        region
            .set_block(
                [0, 1, 0],
                &parse_block(
                    "minecraft:big_dripleaf[facing=north,tilt=unstable,waterlogged=false]",
                ),
            )
            .unwrap();
        let mut powered_ticks = DynamicBlockTicks::bootstrap(&region);
        powered_ticks.run_due(&mut region, 10);
        let reset = block_at(&region, [0, 1, 0]).expect("powered dripleaf remains");
        assert_eq!(big_dripleaf_tilt(reset), "none");
    }

    #[test]
    fn entity_standing_on_big_dripleaf_triggers_unstable_tilt() {
        let mut region = region_with_shape([1, 2, 1]);
        region
            .set_block([0, 0, 0], &parse_block("minecraft:dirt"))
            .unwrap();
        region
            .set_block(
                [0, 1, 0],
                &parse_block("minecraft:big_dripleaf[facing=north,tilt=none,waterlogged=false]"),
            )
            .unwrap();

        let mut block_ticks = DynamicBlockTicks::default();
        maybe_trigger_big_dripleaf(
            &mut region,
            &mut block_ticks,
            Vec3d::new(0.5, 1.0 + 15.0 / 16.0, 0.5),
            VERIFY_DEFAULT_WIDTH,
            true,
            1,
        );
        let block = block_at(&region, [0, 1, 0]).expect("big dripleaf remains");
        assert_eq!(big_dripleaf_tilt(block), "unstable");
        assert!(
            block_ticks
                .big_dripleaf_tilts
                .get(&(1 + BIG_DRIPLEAF_UNSTABLE_TICK_DELAY))
                .map(|positions| positions.contains(&[0, 1, 0]))
                .unwrap_or(false)
        );
    }

    #[test]
    fn entity_loses_ground_support_when_big_dripleaf_is_fully_tilted() {
        let mut region = region_with_shape([1, 2, 1]);
        region
            .set_block([0, 0, 0], &parse_block("minecraft:dirt"))
            .unwrap();
        region
            .set_block(
                [0, 1, 0],
                &parse_block("minecraft:big_dripleaf[facing=north,tilt=full,waterlogged=false]"),
            )
            .unwrap();
        let world = LoadedSchematic {
            name: "full-big-dripleaf-support-loss".to_string(),
            region,
            approximate_collision_blocks: Vec::new(),
        };
        let rows = simulate(
            &world,
            &VerifyCommand {
                input: std::path::PathBuf::from("full-big-dripleaf-support-loss.litematic"),
                out: std::path::PathBuf::from("artifacts/test"),
                target_speed: 0.0,
                ticks: 1,
                inspect_tick: Some(1),
                start_x: 0.5,
                start_y: 1.0 + 15.0 / 16.0,
                start_z: 0.5,
                start_vx: 0.0,
                start_vy: 0.0,
                start_vz: 0.0,
                start_on_ground: true,
                width: VERIFY_DEFAULT_WIDTH,
                height: VERIFY_DEFAULT_HEIGHT,
                entity_id_mod4: 3,
                initial_tick_count: 0,
                entity_rng_seed: None,
                entity_uuid: None,
                bootstrap_fluids: false,
                entity_kind: VerifyEntityKind::Item,
                no_ai: false,
                no_gravity: false,
                fire_immune: false,
                start_fire_ticks: 0,
                item_health: None,
            },
        );
        assert!(!rows[1].on_ground);
        assert!(rows[1].y < 1.0 + 15.0 / 16.0);
    }

    #[test]
    fn carpet_and_snow_layers_match_expected_vertical_collision_height() {
        for (block_id, expected_delta_y) in [
            ("minecraft:white_carpet", -15.0 / 16.0),
            ("minecraft:snow[layers=4]", -10.0 / 16.0),
        ] {
            let mut region = region_with_shape([1, 1, 1]);
            region.set_block([0, 0, 0], &parse_block(block_id)).unwrap();
            let move_result = move_entity(
                &region,
                Vec3d::new(0.5, 1.0, 0.5),
                Vec3d::new(0.0, -1.0, 0.0),
                0.25,
                0.25,
            );
            assert!(
                move_result.collided_y,
                "expected vertical collision for {block_id}"
            );
            assert!(
                (move_result.delta.y - expected_delta_y).abs() < 1.0e-12,
                "unexpected vertical delta for {block_id}: {}",
                move_result.delta.y
            );
        }
    }

    #[test]
    fn pointed_dripstone_tip_does_not_collide_like_a_full_block() {
        let mut region = region_with_shape([1, 1, 1]);
        region
            .set_block(
                [0, 0, 0],
                &parse_block(
                    "minecraft:pointed_dripstone[thickness=tip,vertical_direction=up,waterlogged=false]",
                ),
            )
            .unwrap();
        let move_result = move_entity(
            &region,
            Vec3d::new(0.5, 1.0, 0.5),
            Vec3d::new(0.0, -1.0, 0.0),
            0.25,
            0.25,
        );
        assert!(move_result.collided_y);
        assert!((move_result.delta.y + 5.0 / 16.0).abs() < 1.0e-12);
    }

    #[test]
    fn cactus_pot_and_chest_match_expected_horizontal_collision_width() {
        for (block_id, expected_delta_x) in [
            ("minecraft:cactus[age=0]", 7.0 / 16.0),
            (
                "minecraft:decorated_pot[facing=north,cracked=false,waterlogged=false]",
                7.0 / 16.0,
            ),
            (
                "minecraft:chest[facing=north,type=single,waterlogged=false]",
                7.0 / 16.0,
            ),
            (
                "minecraft:trapped_chest[facing=north,type=single,waterlogged=false]",
                7.0 / 16.0,
            ),
            ("minecraft:barrel[facing=north,open=false]", 6.0 / 16.0),
        ] {
            let mut region = region_with_shape([3, 2, 1]);
            region.set_block([1, 0, 0], &parse_block(block_id)).unwrap();
            let move_result = move_entity(
                &region,
                Vec3d::new(0.5, 0.0, 0.5),
                Vec3d::new(1.0, 0.0, 0.0),
                0.25,
                0.25,
            );
            assert!(
                move_result.collided_x,
                "expected horizontal collision for {block_id}"
            );
            assert!(
                (move_result.delta.x - expected_delta_x).abs() < 1.0e-12,
                "unexpected horizontal delta for {block_id}: {}",
                move_result.delta.x
            );
        }
    }

    #[test]
    fn cactus_survival_rules_match_guardian() {
        let mut supported = region_with_shape([1, 2, 1]);
        supported
            .set_block([0, 0, 0], &parse_block("minecraft:sand"))
            .unwrap();
        supported
            .set_block([0, 1, 0], &parse_block("minecraft:cactus[age=0]"))
            .unwrap();
        run_world_ticks(&mut supported, 1);
        assert_eq!(
            block_full_id(block_at(&supported, [0, 1, 0]).expect("supported cactus remains")),
            "minecraft:cactus[age=0]"
        );

        let unsupported_floor = region_with_shape([1, 2, 1]);
        let cactus = parse_block("minecraft:cactus[age=0]");
        assert!(!cactus_can_survive(&unsupported_floor, [0, 1, 0]));

        let mut snapshot = region_with_shape([1, 2, 1]);
        snapshot.set_block([0, 1, 0], &cactus).unwrap();
        run_world_ticks(&mut snapshot, 3);
        assert_eq!(
            block_full_id(
                block_at(&snapshot, [0, 1, 0])
                    .expect("unsupported cactus persists through three snapshot ticks")
            ),
            "minecraft:cactus[age=0]"
        );

        let mut horizontal_blocked = region_with_shape([2, 2, 1]);
        horizontal_blocked
            .set_block([0, 0, 0], &parse_block("minecraft:sand"))
            .unwrap();
        horizontal_blocked.set_block([0, 1, 0], &cactus).unwrap();
        horizontal_blocked
            .set_block([1, 1, 0], &parse_block("minecraft:stone"))
            .unwrap();
        assert!(!cactus_can_survive(&horizontal_blocked, [0, 1, 0]));

        let mut water_above = region_with_shape([1, 3, 1]);
        water_above
            .set_block([0, 0, 0], &parse_block("minecraft:sand"))
            .unwrap();
        water_above.set_block([0, 1, 0], &cactus).unwrap();
        water_above
            .set_block([0, 2, 0], &parse_block("minecraft:water"))
            .unwrap();
        assert!(!cactus_can_survive(&water_above, [0, 1, 0]));

        let mut scheduled_break = region_with_shape([1, 2, 1]);
        scheduled_break.set_block([0, 1, 0], &cactus).unwrap();
        let mut block_ticks = DynamicBlockTicks::default();
        block_ticks.schedule(CACTUS_TICK_DELAY, [0, 1, 0]);
        block_ticks.run_due(&mut scheduled_break, CACTUS_TICK_DELAY);
        assert_eq!(
            block_full_id(
                block_at(&scheduled_break, [0, 1, 0]).expect("scheduled cactus break yields air")
            ),
            "minecraft:air"
        );
    }

    #[test]
    fn ladder_support_face_rules_match_guardian() {
        let mut supported = region_with_shape([1, 1, 2]);
        supported
            .set_block([0, 0, 1], &parse_block("minecraft:stone"))
            .unwrap();
        let supported_ladder = parse_block("minecraft:ladder[facing=north,waterlogged=false]");
        assert!(ladder_can_survive(&supported, [0, 0, 0], &supported_ladder));

        let unsupported = region_with_shape([1, 1, 1]);
        let unsupported_ladder = parse_block("minecraft:ladder[facing=north,waterlogged=false]");
        assert!(!ladder_can_survive(
            &unsupported,
            [0, 0, 0],
            &unsupported_ladder
        ));
    }

    #[test]
    fn scheduled_unsupported_ladder_breaks_when_its_tick_is_due() {
        let mut region = region_with_shape([1, 1, 1]);
        region
            .set_block(
                [0, 0, 0],
                &parse_block("minecraft:ladder[facing=north,waterlogged=false]"),
            )
            .unwrap();
        let mut block_ticks = DynamicBlockTicks::default();
        block_ticks.schedule(LADDER_TICK_DELAY, [0, 0, 0]);
        block_ticks.run_due(&mut region, LADDER_TICK_DELAY);
        assert_eq!(
            block_full_id(block_at(&region, [0, 0, 0]).expect("scheduled ladder break yields air")),
            "minecraft:air"
        );
    }

    #[test]
    fn wall_skulls_block_entity_motion() {
        for (block_id, expected_delta_x) in [
            ("minecraft:skeleton_wall_skull[facing=east]", 3.0 / 8.0),
            ("minecraft:player_wall_head[facing=east]", 3.0 / 8.0),
        ] {
            let mut region = region_with_shape([3, 2, 1]);
            region.set_block([1, 0, 0], &parse_block(block_id)).unwrap();
            let move_result = move_entity(
                &region,
                Vec3d::new(0.5, 0.375, 0.5),
                Vec3d::new(1.0, 0.0, 0.0),
                0.25,
                0.25,
            );
            assert!(move_result.collided_x, "expected collision for {block_id}");
            assert!(
                (move_result.delta.x - expected_delta_x).abs() < 1.0e-12,
                "unexpected horizontal delta for {block_id}: {}",
                move_result.delta.x
            );
        }
    }

    #[test]
    fn standing_skulls_and_heads_block_entity_motion() {
        for (block_id, expected_delta_x) in [
            ("minecraft:skeleton_skull[rotation=0]", 5.0 / 8.0),
            ("minecraft:player_head[rotation=0]", 5.0 / 8.0),
            ("minecraft:piglin_head[rotation=0]", 9.0 / 16.0),
        ] {
            let mut region = region_with_shape([3, 2, 1]);
            region.set_block([1, 0, 0], &parse_block(block_id)).unwrap();
            let move_result = move_entity(
                &region,
                Vec3d::new(0.5, 0.0, 0.5),
                Vec3d::new(1.0, 0.0, 0.0),
                0.25,
                0.25,
            );
            assert!(move_result.collided_x, "expected collision for {block_id}");
            assert!(
                (move_result.delta.x - expected_delta_x).abs() < 1.0e-12,
                "unexpected horizontal delta for {block_id}: {}",
                move_result.delta.x
            );
        }
    }

    #[test]
    fn advanced_collision_rows_match_vanilla_mixed_one_tick_trace() {
        const POSITION_EPS: f64 = 5.0e-8;
        const VELOCITY_EPS: f64 = 5.0e-8;

        let world = advanced_collision_probe_world();
        for (
            start_x,
            start_z,
            expected_x,
            expected_y,
            expected_z,
            expected_vx,
            expected_vy,
            expected_vz,
            entity_rng_seed,
        ) in [
            (
                3.5,
                0.5,
                4.5,
                1.16,
                0.5,
                0.9800000190734863,
                -0.03920000076293945,
                0.0,
                None,
            ),
            (
                7.5,
                0.5,
                8.5,
                1.16,
                0.5,
                0.9800000190734863,
                -0.03920000076293945,
                0.0,
                None,
            ),
            (
                11.5,
                0.5,
                12.125,
                1.16,
                0.5,
                0.0,
                -0.03920000076293945,
                0.0,
                None,
            ),
            (
                3.5,
                4.5,
                4.125,
                1.16,
                4.5,
                0.0,
                -0.03920000076293945,
                0.0,
                None,
            ),
            (
                7.5,
                4.5,
                8.25,
                1.17,
                4.38218011707067,
                0.7350000143051147,
                -0.02940000057220459,
                -0.11546348751797453,
                Some(46_458_538_216_644),
            ),
            (
                11.5,
                4.5,
                11.9375,
                1.16,
                4.5,
                0.0,
                -0.03920000076293945,
                0.0,
                None,
            ),
            (
                3.5,
                8.5,
                3.9375,
                1.16,
                8.5,
                0.0,
                -0.03920000076293945,
                0.0,
                None,
            ),
            (
                7.5,
                8.5,
                7.9375,
                1.16,
                8.5,
                0.0,
                -0.03920000076293945,
                0.0,
                None,
            ),
            (
                11.5,
                8.5,
                11.875,
                1.16,
                8.5,
                0.0,
                -0.03920000076293945,
                0.0,
                None,
            ),
        ] {
            let command = VerifyCommand {
                input: std::path::PathBuf::from("advanced-collision-probe.litematic"),
                out: std::path::PathBuf::from("artifacts/test"),
                target_speed: 0.0,
                ticks: 1,
                inspect_tick: Some(1),
                start_x,
                start_y: 1.2,
                start_z,
                start_vx: 1.0,
                start_vy: 0.0,
                start_vz: 0.0,
                start_on_ground: false,
                width: VERIFY_DEFAULT_WIDTH,
                height: VERIFY_DEFAULT_HEIGHT,
                entity_id_mod4: 0,
                initial_tick_count: 0,
                entity_rng_seed,
                entity_uuid: None,
                bootstrap_fluids: false,
                entity_kind: VerifyEntityKind::Item,
                no_ai: false,

                no_gravity: false,
                fire_immune: false,
                start_fire_ticks: 0,
                item_health: None,
            };

            let rows = simulate(&world, &command);
            let tick = &rows[1];
            assert!(
                (tick.x - expected_x).abs() < POSITION_EPS,
                "unexpected x for start ({start_x}, {start_z}): {}",
                tick.x
            );
            assert!(
                (tick.y - expected_y).abs() < POSITION_EPS,
                "unexpected y for start ({start_x}, {start_z}): {}",
                tick.y
            );
            assert!(
                (tick.z - expected_z).abs() < POSITION_EPS,
                "unexpected z for start ({start_x}, {start_z}): {}",
                tick.z
            );
            assert!(
                (tick.vx - expected_vx).abs() < VELOCITY_EPS,
                "unexpected vx for start ({start_x}, {start_z}): {}",
                tick.vx
            );
            assert!(
                (tick.vy - expected_vy).abs() < VELOCITY_EPS,
                "unexpected vy for start ({start_x}, {start_z}): {}",
                tick.vy
            );
            assert!(
                (tick.vz - expected_vz).abs() < VELOCITY_EPS,
                "unexpected vz for start ({start_x}, {start_z}): {}",
                tick.vz
            );
        }
    }

    #[test]
    fn verify_matches_vanilla_button_probe_rows() {
        const POSITION_EPS: f64 = 5.0e-8;
        const VELOCITY_EPS: f64 = 5.0e-8;

        let world = button_probe_world();
        for (
            start_x,
            start_y,
            start_z,
            start_vx,
            start_vy,
            expected_x,
            expected_y,
            expected_z,
            expected_vx,
            expected_vy,
            expected_vz,
            expected_on_ground,
            entity_rng_seed,
        ) in [
            (
                4.5,
                1.25,
                1.5,
                1.0,
                0.0,
                5.25,
                1.22,
                1.255622774362564,
                0.7350000143051147,
                -0.02940000057220459,
                -0.23948968578581287,
                false,
                Some(167_617_804_029_124),
            ),
            (
                4.5,
                1.25,
                2.5,
                1.0,
                0.0,
                5.25,
                1.22,
                2.668023735284805,
                0.7350000143051147,
                -0.02940000057220459,
                0.1646632637839076,
                false,
                Some(67_922_704_387_268),
            ),
            (
                5.5, 2.0, 3.375, 0.0, -1.0, 5.5, 1.0, 3.375, 0.0, 0.0, 0.0, true, None,
            ),
            (
                5.5, 2.0, 4.375, 0.0, -1.0, 5.5, 1.0, 4.375, 0.0, 0.0, 0.0, true, None,
            ),
            (
                5.5, 1.0, 5.375, 0.0, 1.0, 5.5, 1.75, 5.375, 0.0, 0.0, 0.0, false, None,
            ),
            (
                5.5, 1.0, 6.375, 0.0, 1.0, 5.5, 1.75, 6.375, 0.0, 0.0, 0.0, false, None,
            ),
        ] {
            let command = VerifyCommand {
                input: std::path::PathBuf::from("button-probe.litematic"),
                out: std::path::PathBuf::from("artifacts/test"),
                target_speed: 0.0,
                ticks: 1,
                inspect_tick: Some(1),
                start_x,
                start_y,
                start_z,
                start_vx,
                start_vy,
                start_vz: 0.0,
                start_on_ground: false,
                width: VERIFY_DEFAULT_WIDTH,
                height: VERIFY_DEFAULT_HEIGHT,
                entity_id_mod4: 0,
                initial_tick_count: 0,
                entity_rng_seed,
                entity_uuid: None,
                bootstrap_fluids: false,
                entity_kind: VerifyEntityKind::Item,
                no_ai: false,

                no_gravity: false,
                fire_immune: false,
                start_fire_ticks: 0,
                item_health: None,
            };

            let rows = simulate(&world, &command);
            let tick = &rows[1];
            assert!(
                (tick.x - expected_x).abs() < POSITION_EPS,
                "unexpected x for start ({start_x}, {start_y}, {start_z}): {}",
                tick.x
            );
            assert!(
                (tick.y - expected_y).abs() < POSITION_EPS,
                "unexpected y for start ({start_x}, {start_y}, {start_z}): {}",
                tick.y
            );
            assert!(
                (tick.z - expected_z).abs() < POSITION_EPS,
                "unexpected z for start ({start_x}, {start_y}, {start_z}): {}",
                tick.z
            );
            assert!(
                (tick.vx - expected_vx).abs() < VELOCITY_EPS,
                "unexpected vx for start ({start_x}, {start_y}, {start_z}): {}",
                tick.vx
            );
            assert!(
                (tick.vy - expected_vy).abs() < VELOCITY_EPS,
                "unexpected vy for start ({start_x}, {start_y}, {start_z}): {}",
                tick.vy
            );
            assert!(
                (tick.vz - expected_vz).abs() < VELOCITY_EPS,
                "unexpected vz for start ({start_x}, {start_y}, {start_z}): {}",
                tick.vz
            );
            assert_eq!(tick.on_ground, expected_on_ground);
        }
    }

    #[test]
    fn verify_matches_vanilla_hopper_horizontal_probe_rows() {
        const POSITION_EPS: f64 = 5.0e-8;
        const VELOCITY_EPS: f64 = 5.0e-8;

        let world = hopper_probe_world();
        for (
            start_x,
            start_y,
            start_z,
            expected_x,
            expected_y,
            expected_z,
            expected_vx,
            expected_vy,
            expected_vz,
            expected_on_ground,
        ) in [
            (
                4.5,
                1.25,
                0.5,
                5.125,
                1.21,
                0.5,
                0.0,
                -0.03920000076293945,
                0.0,
                false,
            ),
            (
                4.5,
                1.25,
                2.5,
                5.125,
                1.21,
                2.5,
                0.0,
                -0.03920000076293945,
                0.0,
                false,
            ),
        ] {
            let command = VerifyCommand {
                input: std::path::PathBuf::from("hopper-probe.litematic"),
                out: std::path::PathBuf::from("artifacts/test"),
                target_speed: 0.0,
                ticks: 1,
                inspect_tick: Some(1),
                start_x,
                start_y,
                start_z,
                start_vx: 1.0,
                start_vy: 0.0,
                start_vz: 0.0,
                start_on_ground: false,
                width: VERIFY_DEFAULT_WIDTH,
                height: VERIFY_DEFAULT_HEIGHT,
                entity_id_mod4: 0,
                initial_tick_count: 0,
                entity_rng_seed: None,
                entity_uuid: None,
                bootstrap_fluids: false,
                entity_kind: VerifyEntityKind::Item,
                no_ai: false,

                no_gravity: false,
                fire_immune: false,
                start_fire_ticks: 0,
                item_health: None,
            };

            let rows = simulate(&world, &command);
            let tick = &rows[1];
            assert!(
                tick.alive,
                "expected alive row for start ({start_x}, {start_y}, {start_z})"
            );
            assert!(tick.removed_by.is_empty());
            assert!((tick.x - expected_x).abs() < POSITION_EPS);
            assert!((tick.y - expected_y).abs() < POSITION_EPS);
            assert!((tick.z - expected_z).abs() < POSITION_EPS);
            assert!((tick.vx - expected_vx).abs() < VELOCITY_EPS);
            assert!((tick.vy - expected_vy).abs() < VELOCITY_EPS);
            assert!((tick.vz - expected_vz).abs() < VELOCITY_EPS);
            assert_eq!(tick.on_ground, expected_on_ground);
        }
    }

    #[test]
    fn verify_matches_vanilla_hopper_collection_paths() {
        let world = hopper_probe_world();
        for (start_y, start_z, start_vy, expected_removed_by) in [
            (2.0, 4.5, -1.0, "hopperEntityInside"),
            (1.0, 6.5, 1.0, "hopperTickSuck"),
        ] {
            let command = VerifyCommand {
                input: std::path::PathBuf::from("hopper-probe.litematic"),
                out: std::path::PathBuf::from("artifacts/test"),
                target_speed: 0.0,
                ticks: 1,
                inspect_tick: Some(1),
                start_x: 5.5,
                start_y,
                start_z,
                start_vx: 0.0,
                start_vy,
                start_vz: 0.0,
                start_on_ground: false,
                width: VERIFY_DEFAULT_WIDTH,
                height: VERIFY_DEFAULT_HEIGHT,
                entity_id_mod4: 0,
                initial_tick_count: 0,
                entity_rng_seed: None,
                entity_uuid: None,
                bootstrap_fluids: false,
                entity_kind: VerifyEntityKind::Item,
                no_ai: false,

                no_gravity: false,
                fire_immune: false,
                start_fire_ticks: 0,
                item_health: None,
            };

            let rows = simulate(&world, &command);
            let tick = &rows[1];
            assert!(!tick.alive, "expected hopper collection for z={start_z}");
            assert_eq!(tick.removed_by, expected_removed_by);
        }
    }

    #[test]
    fn generic_entities_are_not_collected_by_hoppers() {
        let world = hopper_probe_world();
        let command = VerifyCommand {
            input: std::path::PathBuf::from("hopper-probe.litematic"),
            out: std::path::PathBuf::from("artifacts/test"),
            target_speed: 0.0,
            ticks: 1,
            inspect_tick: Some(1),
            start_x: 5.5,
            start_y: 2.0,
            start_z: 4.5,
            start_vx: 0.0,
            start_vy: -1.0,
            start_vz: 0.0,
            start_on_ground: false,
            width: VERIFY_DEFAULT_WIDTH,
            height: VERIFY_DEFAULT_HEIGHT,
            entity_id_mod4: 0,
            initial_tick_count: 0,
            entity_rng_seed: None,
            entity_uuid: None,
            bootstrap_fluids: false,
            entity_kind: VerifyEntityKind::Generic,
            no_ai: false,

            no_gravity: false,
            fire_immune: false,
            start_fire_ticks: 0,
            item_health: None,
        };

        let rows = simulate(&world, &command);
        let tick = &rows[1];
        assert!(tick.alive);
        assert!(tick.removed_by.is_empty());
    }

    #[test]
    fn hopper_transfer_cooldown_two_blocks_same_tick_suction() {
        let mut world = hopper_probe_world();
        world
            .region
            .block_entities
            .get_mut(&[5, 1, 4])
            .expect("hopper block entity")
            .tags
            .insert("TransferCooldown".to_string(), Value::Int(2));

        let command = VerifyCommand {
            input: std::path::PathBuf::from("hopper-probe.litematic"),
            out: std::path::PathBuf::from("artifacts/test"),
            target_speed: 0.0,
            ticks: 1,
            inspect_tick: Some(1),
            start_x: 5.5,
            start_y: 2.0,
            start_z: 4.5,
            start_vx: 0.0,
            start_vy: -1.0,
            start_vz: 0.0,
            start_on_ground: false,
            width: VERIFY_DEFAULT_WIDTH,
            height: VERIFY_DEFAULT_HEIGHT,
            entity_id_mod4: 0,
            initial_tick_count: 0,
            entity_rng_seed: None,
            entity_uuid: None,
            bootstrap_fluids: false,
            entity_kind: VerifyEntityKind::Item,
            no_ai: false,

            no_gravity: false,
            fire_immune: false,
            start_fire_ticks: 0,
            item_health: None,
        };

        let rows = simulate(&world, &command);
        let tick = &rows[1];
        assert!(
            tick.alive,
            "cooldown=2 should still block hopper suction this tick"
        );
        assert!(tick.removed_by.is_empty());
    }

    #[test]
    fn hopper_transfer_cooldown_one_allows_post_tick_suction() {
        let mut world = hopper_probe_world();
        world
            .region
            .block_entities
            .get_mut(&[5, 1, 4])
            .expect("hopper block entity")
            .tags
            .insert("TransferCooldown".to_string(), Value::Int(1));

        let command = VerifyCommand {
            input: std::path::PathBuf::from("hopper-probe.litematic"),
            out: std::path::PathBuf::from("artifacts/test"),
            target_speed: 0.0,
            ticks: 1,
            inspect_tick: Some(1),
            start_x: 5.5,
            start_y: 2.0,
            start_z: 4.5,
            start_vx: 0.0,
            start_vy: -1.0,
            start_vz: 0.0,
            start_on_ground: false,
            width: VERIFY_DEFAULT_WIDTH,
            height: VERIFY_DEFAULT_HEIGHT,
            entity_id_mod4: 0,
            initial_tick_count: 0,
            entity_rng_seed: None,
            entity_uuid: None,
            bootstrap_fluids: false,
            entity_kind: VerifyEntityKind::Item,
            no_ai: false,

            no_gravity: false,
            fire_immune: false,
            start_fire_ticks: 0,
            item_health: None,
        };

        let rows = simulate(&world, &command);
        let tick = &rows[1];
        assert!(
            !tick.alive,
            "cooldown=1 should decrement and allow hopper suction"
        );
        assert_eq!(tick.removed_by, "hopperTickSuck");
    }

    #[test]
    fn hopper_with_container_above_does_not_suck_air_item() {
        let world = hopper_container_probe_world();
        let command = VerifyCommand {
            input: std::path::PathBuf::from("hopper-container-probe.litematic"),
            out: std::path::PathBuf::from("artifacts/test"),
            target_speed: 0.0,
            ticks: 1,
            inspect_tick: Some(1),
            start_x: 1.5,
            start_y: 2.75,
            start_z: 0.5,
            start_vx: 0.0,
            start_vy: 0.0,
            start_vz: 0.0,
            start_on_ground: false,
            width: VERIFY_DEFAULT_WIDTH,
            height: VERIFY_DEFAULT_HEIGHT,
            entity_id_mod4: 0,
            initial_tick_count: 0,
            entity_rng_seed: None,
            entity_uuid: None,
            bootstrap_fluids: false,
            entity_kind: VerifyEntityKind::Item,
            no_ai: false,

            no_gravity: false,
            fire_immune: false,
            start_fire_ticks: 0,
            item_health: None,
        };

        let rows = simulate(&world, &command);
        let tick = &rows[1];
        assert!(
            tick.alive,
            "source container above hopper should block air-item suction"
        );
        assert!(tick.removed_by.is_empty());
    }

    #[test]
    fn end_rod_blocks_entity_motion() {
        let mut region = region_with_shape([3, 2, 1]);
        region
            .set_block([1, 0, 0], &parse_block("minecraft:end_rod[facing=up]"))
            .unwrap();
        let move_result = move_entity(
            &region,
            Vec3d::new(0.5, 0.0, 0.5),
            Vec3d::new(1.0, 0.0, 0.0),
            0.25,
            0.25,
        );
        assert!(move_result.collided_x);
        assert!((move_result.delta.x - 0.75).abs() < 1.0e-12);
    }

    #[test]
    fn buttons_do_not_block_entity_motion() {
        for (block_id, start_pos, velocity) in [
            (
                "minecraft:stone_button[face=wall,facing=east,powered=false]",
                Vec3d::new(0.5, 0.25, 0.5),
                Vec3d::new(1.0, 0.0, 0.0),
            ),
            (
                "minecraft:stone_button[face=floor,facing=north,powered=false]",
                Vec3d::new(0.5, 1.0, 0.375),
                Vec3d::new(0.0, -1.0, 0.0),
            ),
            (
                "minecraft:stone_button[face=ceiling,facing=north,powered=false]",
                Vec3d::new(0.5, 0.0, 0.375),
                Vec3d::new(0.0, 1.0, 0.0),
            ),
        ] {
            let mut region = region_with_shape([3, 2, 1]);
            region.set_block([1, 0, 0], &parse_block(block_id)).unwrap();
            let move_result = move_entity(&region, start_pos, velocity, 0.25, 0.25);
            assert!(
                !move_result.collided_x && !move_result.collided_y && !move_result.collided_z,
                "unexpected collision for {block_id}"
            );
            assert!(
                (move_result.delta.x - velocity.x).abs() < 1.0e-12
                    && (move_result.delta.y - velocity.y).abs() < 1.0e-12
                    && (move_result.delta.z - velocity.z).abs() < 1.0e-12,
                "unexpected movement delta for {block_id}: {:?}",
                move_result.delta
            );
        }
    }

    #[test]
    fn pressure_plate_does_not_block_vertical_motion() {
        let mut region = region_with_shape([1, 1, 1]);
        region
            .set_block(
                [0, 0, 0],
                &parse_block("minecraft:oak_pressure_plate[powered=true]"),
            )
            .unwrap();
        let move_result = move_entity(
            &region,
            Vec3d::new(0.5, 1.0, 0.5),
            Vec3d::new(0.0, -1.0, 0.0),
            0.25,
            0.25,
        );
        assert!(!move_result.collided_y);
        assert!((move_result.delta.y + 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn partial_collision_rows_match_vanilla_two_tick_trace() {
        const POSITION_EPS: f64 = 5.0e-8;
        const VELOCITY_EPS: f64 = 5.0e-8;
        const EXPECTED_Y: f64 = 1.0807999992370605;
        const EXPECTED_VY: f64 = -0.0776160022583008;

        for (block_id, expected_x, expected_vx) in [
            (
                "minecraft:oak_wall_hanging_sign[facing=east,waterlogged=false]",
                2.4800000190734863,
                0.9604000373840336,
            ),
            (
                "minecraft:oak_pressure_plate[powered=false]",
                2.4800000190734863,
                0.9604000373840336,
            ),
            ("minecraft:end_rod[facing=up]", 1.25, 0.0),
            (
                "minecraft:iron_chain[axis=y,waterlogged=false]",
                1.28125,
                0.0,
            ),
            (
                "minecraft:lightning_rod[facing=up,powered=false,waterlogged=false]",
                1.25,
                0.0,
            ),
        ] {
            let world = partial_collision_probe_world(block_id);
            let command = VerifyCommand {
                input: std::path::PathBuf::from("partial-collision-probe.litematic"),
                out: std::path::PathBuf::from("artifacts/test"),
                target_speed: 0.0,
                ticks: 2,
                inspect_tick: Some(2),
                start_x: 0.5,
                start_y: 1.2,
                start_z: 1.5,
                start_vx: 1.0,
                start_vy: 0.0,
                start_vz: 0.0,
                start_on_ground: false,
                width: VERIFY_DEFAULT_WIDTH,
                height: VERIFY_DEFAULT_HEIGHT,
                entity_id_mod4: 0,
                initial_tick_count: 0,
                entity_rng_seed: None,
                entity_uuid: None,
                bootstrap_fluids: false,
                entity_kind: VerifyEntityKind::Item,
                no_ai: false,

                no_gravity: false,
                fire_immune: false,
                start_fire_ticks: 0,
                item_health: None,
            };

            let rows = simulate(&world, &command);
            let tick = &rows[2];
            assert!(
                (tick.x - expected_x).abs() < POSITION_EPS,
                "unexpected x for {block_id}: {}",
                tick.x
            );
            assert!(
                (tick.y - EXPECTED_Y).abs() < POSITION_EPS,
                "unexpected y for {block_id}: {}",
                tick.y
            );
            assert!(
                (tick.vx - expected_vx).abs() < VELOCITY_EPS,
                "unexpected vx for {block_id}: {}",
                tick.vx
            );
            assert!(
                (tick.vy - EXPECTED_VY).abs() < VELOCITY_EPS,
                "unexpected vy for {block_id}: {}",
                tick.vy
            );
            assert!(!tick.on_ground, "unexpected on_ground for {block_id}");
        }
    }

    #[test]
    fn stair_collision_rows_match_vanilla_two_tick_trace() {
        const POSITION_EPS: f64 = 5.0e-8;
        const VELOCITY_EPS: f64 = 5.0e-8;
        const EXPECTED_Y: f64 = 1.0807999992370605;
        const EXPECTED_VY: f64 = -0.0776160022583008;

        for (block_id, expected_x) in [
            (
                "minecraft:oak_stairs[facing=east,half=bottom,shape=straight,waterlogged=false]",
                0.875,
            ),
            (
                "minecraft:oak_stairs[facing=east,half=bottom,shape=outer_left,waterlogged=false]",
                0.875,
            ),
            (
                "minecraft:oak_stairs[facing=east,half=bottom,shape=inner_left,waterlogged=false]",
                0.875,
            ),
            (
                "minecraft:oak_stairs[facing=east,half=top,shape=straight,waterlogged=false]",
                1.375,
            ),
            (
                "minecraft:oak_stairs[facing=east,half=top,shape=inner_left,waterlogged=false]",
                0.875,
            ),
        ] {
            let world = partial_collision_probe_world(block_id);
            let command = VerifyCommand {
                input: std::path::PathBuf::from("stair-collision-probe.litematic"),
                out: std::path::PathBuf::from("artifacts/test"),
                target_speed: 0.0,
                ticks: 2,
                inspect_tick: Some(2),
                start_x: 0.5,
                start_y: 1.2,
                start_z: 1.5,
                start_vx: 1.0,
                start_vy: 0.0,
                start_vz: 0.0,
                start_on_ground: false,
                width: VERIFY_DEFAULT_WIDTH,
                height: VERIFY_DEFAULT_HEIGHT,
                entity_id_mod4: 0,
                initial_tick_count: 0,
                entity_rng_seed: None,
                entity_uuid: None,
                bootstrap_fluids: false,
                entity_kind: VerifyEntityKind::Item,
                no_ai: false,

                no_gravity: false,
                fire_immune: false,
                start_fire_ticks: 0,
                item_health: None,
            };

            let rows = simulate(&world, &command);
            let tick = &rows[2];
            assert!(
                (tick.x - expected_x).abs() < POSITION_EPS,
                "unexpected x for {block_id}: {}",
                tick.x
            );
            assert!(
                (tick.y - EXPECTED_Y).abs() < POSITION_EPS,
                "unexpected y for {block_id}: {}",
                tick.y
            );
            assert!(
                tick.vx.abs() < VELOCITY_EPS,
                "unexpected vx for {block_id}: {}",
                tick.vx
            );
            assert!(
                (tick.vy - EXPECTED_VY).abs() < VELOCITY_EPS,
                "unexpected vy for {block_id}: {}",
                tick.vy
            );
            assert!(!tick.on_ground, "unexpected on_ground for {block_id}");
        }
    }
    #[test]
    fn double_slab_cannot_be_waterlogged_or_replaced_by_spread_to() {
        let mut region = region_with_shape([1, 1, 1]);
        let slab = parse_block("minecraft:oak_slab[type=double]");
        region.set_block([0, 0, 0], &slab).unwrap();
        let mut fluid_ticks = DynamicFluidTicks::default();
        let mut block_ticks = DynamicBlockTicks::default();
        fluid_ticks.spread_to(
            &mut region,
            &mut block_ticks,
            [0, 0, 0],
            &slab,
            DynamicWaterState {
                amount: 8,
                falling: false,
            },
            0,
        );
        let block = block_at(&region, [0, 0, 0]).expect("double slab remains");
        assert_eq!(block_full_id(block), block_full_id(&slab));
        assert!(!can_hold_specific_fluid(
            &slab,
            DynamicWaterState {
                amount: 8,
                falling: false,
            }
        ));
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
            entity_rng_seed: None,
            entity_uuid: None,
            bootstrap_fluids: true,
            entity_kind: VerifyEntityKind::Item,
            no_ai: false,

            no_gravity: false,
            fire_immune: false,
            start_fire_ticks: 0,
            item_health: None,
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
    fn verify_matches_vanilla_connected_bars_collision_tick() {
        let mut region = region_with_shape([7, 4, 3]);
        for x in 0..=6 {
            for z in 0..=2 {
                region
                    .set_block([x, 0, z], &parse_block("minecraft:smooth_stone"))
                    .unwrap();
            }
        }
        region
            .set_block(
                [3, 1, 1],
                &parse_block("minecraft:iron_bars[north=true,south=true,east=false,west=false]"),
            )
            .unwrap();
        let world = LoadedSchematic {
            name: "connected-bars-collision".to_string(),
            region,
            approximate_collision_blocks: Vec::new(),
        };
        let command = VerifyCommand {
            input: std::path::PathBuf::from("connected-bars-collision.litematic"),
            out: std::path::PathBuf::from("artifacts/test"),
            target_speed: 0.0,
            ticks: 1,
            inspect_tick: Some(1),
            start_x: 2.5,
            start_y: 1.2,
            start_z: 1.5,
            start_vx: 1.0,
            start_vy: 0.0,
            start_vz: 0.0,
            start_on_ground: false,
            width: VERIFY_DEFAULT_WIDTH,
            height: VERIFY_DEFAULT_HEIGHT,
            entity_id_mod4: 0,
            initial_tick_count: 0,
            entity_rng_seed: None,
            entity_uuid: None,
            bootstrap_fluids: false,
            entity_kind: VerifyEntityKind::Item,
            no_ai: false,

            no_gravity: false,
            fire_immune: false,
            start_fire_ticks: 0,
            item_health: None,
        };

        let rows = simulate(&world, &command);
        let tick = &rows[1];
        assert!((tick.x - 3.3125).abs() < 1.0e-12);
        assert!((tick.y - 1.16).abs() < 1.0e-12);
        assert!((tick.z - 1.5).abs() < 1.0e-12);
        assert!((tick.vx - 0.0).abs() < 1.0e-12);
        assert!((tick.vy + 0.03920000076293945).abs() < 1.0e-12);
        assert!((tick.vz - 0.0).abs() < 1.0e-12);
        assert!(tick.collided_x);
        assert!(!tick.collided_z);
        assert!(!tick.on_ground);
    }

    #[test]
    fn mixed_sign_lane_matches_vanilla_static_snapshot() {
        let world = mixed_sign_world();
        let mut region = world.region.clone();
        run_fluid_ticks(&mut region, 80);

        let expected = [
            (1, "minecraft:water"),
            (2, "minecraft:oak_sign[rotation=0,waterlogged=false]"),
            (3, "minecraft:air"),
            (4, "minecraft:oak_wall_sign[facing=north,waterlogged=false]"),
            (5, "minecraft:air"),
            (6, "minecraft:oak_fence_gate[facing=north,open=true]"),
            (7, "minecraft:air"),
            (
                8,
                "minecraft:iron_bars[north=false,south=false,east=false,west=false,waterlogged=false]",
            ),
            (9, "minecraft:water"),
            (
                10,
                "minecraft:cobblestone_wall[north=none,south=none,east=none,west=none,up=true,waterlogged=false]",
            ),
            (
                11,
                "minecraft:oak_trapdoor[half=bottom,open=true,facing=north,waterlogged=false]",
            ),
            (
                12,
                "minecraft:oak_stairs[facing=east,half=bottom,shape=straight,waterlogged=false]",
            ),
            (13, "minecraft:air"),
            (14, "minecraft:oak_fence_gate[facing=north,open=false]"),
            (15, "minecraft:water"),
        ];

        for (x, expected_block) in expected {
            let block = block_at(&region, [x, 1, 1]).expect("lane block");
            let normalized_expected = block_full_id(&parse_block(expected_block));
            assert_eq!(
                block_full_id(block),
                normalized_expected,
                "unexpected block at x={x}"
            );
        }
    }

    #[test]
    fn verify_matches_vanilla_mixed_sign_checkpoints() {
        let world = mixed_sign_world();
        let command = VerifyCommand {
            input: std::path::PathBuf::from("mixed-sign.litematic"),
            out: std::path::PathBuf::from("artifacts/test"),
            target_speed: 0.0,
            ticks: 80,
            inspect_tick: Some(80),
            start_x: 2.5,
            start_y: 1.2,
            start_z: 1.5,
            start_vx: 1.0,
            start_vy: 0.0,
            start_vz: 0.0,
            start_on_ground: false,
            width: VERIFY_DEFAULT_WIDTH,
            height: VERIFY_DEFAULT_HEIGHT,
            entity_id_mod4: 0,
            initial_tick_count: 0,
            entity_rng_seed: None,
            entity_uuid: None,
            bootstrap_fluids: true,
            entity_kind: VerifyEntityKind::Item,
            no_ai: false,

            no_gravity: false,
            fire_immune: false,
            start_fire_ticks: 0,
            item_health: None,
        };

        let rows = simulate(&world, &command);
        for (vanilla_tick, sim_tick) in [(12, 11), (40, 39), (80, 79)] {
            let row = &rows[sim_tick];
            assert!(
                (row.x - 8.3125).abs() < 1.0e-12,
                "unexpected x at sim tick {sim_tick} for vanilla tick {vanilla_tick}"
            );
            assert!(
                (row.y - 1.0).abs() < 1.0e-12,
                "unexpected y at sim tick {sim_tick} for vanilla tick {vanilla_tick}"
            );
            assert!(
                (row.z - 1.5).abs() < 1.0e-12,
                "unexpected z at sim tick {sim_tick} for vanilla tick {vanilla_tick}"
            );
            assert!(
                (row.vx - 0.0).abs() < 1.0e-12,
                "unexpected vx at sim tick {sim_tick} for vanilla tick {vanilla_tick}"
            );
            assert!(
                (row.vy + 0.12).abs() < 1.0e-12,
                "unexpected vy at sim tick {sim_tick} for vanilla tick {vanilla_tick}: {}",
                row.vy
            );
            assert!(
                (row.vz - 0.0).abs() < 1.0e-12,
                "unexpected vz at sim tick {sim_tick} for vanilla tick {vanilla_tick}"
            );
        }
    }
    #[test]
    fn verify_matches_vanilla_mixed_sign_trace_prefix() {
        let world = mixed_sign_world();
        let command = VerifyCommand {
            input: std::path::PathBuf::from("mixed-sign.litematic"),
            out: std::path::PathBuf::from("artifacts/test"),
            target_speed: 0.0,
            ticks: 16,
            inspect_tick: Some(16),
            start_x: 2.5,
            start_y: 1.2,
            start_z: 1.5,
            start_vx: 1.0,
            start_vy: 0.0,
            start_vz: 0.0,
            start_on_ground: false,
            width: VERIFY_DEFAULT_WIDTH,
            height: VERIFY_DEFAULT_HEIGHT,
            entity_id_mod4: 0,
            initial_tick_count: 0,
            entity_rng_seed: None,
            entity_uuid: None,
            bootstrap_fluids: true,
            entity_kind: VerifyEntityKind::Item,
            no_ai: false,

            no_gravity: false,
            fire_immune: false,
            start_fire_ticks: 0,
            item_health: None,
        };

        let rows = simulate(&world, &command);
        let expected = [
            (1_usize, 2.5, 1.2, 1.5, 1.0, 0.0, 0.0, false),
            (
                2,
                3.5,
                1.16,
                1.5,
                0.9800000190734863,
                -0.03920000076293945,
                0.0,
                false,
            ),
            (
                3,
                4.47020002822876,
                1.1212999992608,
                1.5,
                0.9507960461692817,
                -0.037926001462550846,
                0.0,
                false,
            ),
            (
                4,
                5.42099607439803,
                1.04337399779824,
                1.5,
                0.9317801433808914,
                -0.07636748291962037,
                0.0,
                false,
            ),
            (
                5,
                6.34345842523126,
                1.0,
                1.5,
                0.885932883175454,
                0.0,
                0.0,
                true,
            ),
            (
                6,
                7.22939130840672,
                1.0,
                1.5,
                0.8508499807960928,
                0.0,
                0.0,
                true,
            ),
            (
                7,
                8.0802412892028,
                1.0,
                1.5,
                0.8171563597750983,
                0.0,
                0.0,
                true,
            ),
            (8, 8.3125, 1.0, 1.5, 0.0, 0.0, 0.0, true),
            (9, 8.3125, 1.0, 1.5, 0.0, 0.0, 0.0, true),
            (10, 8.3125, 1.0, 1.5, 0.0, -0.04, 0.0, true),
            (11, 8.3125, 1.0, 1.5, 0.0, -0.08, 0.0, true),
            (12, 8.3125, 1.0, 1.5, 0.0, -0.12, 0.0, true),
            (13, 8.3125, 1.0, 1.5, 0.0, 0.0, 0.0, true),
            (14, 8.3125, 1.0, 1.5, 0.0, -0.04, 0.0, true),
            (15, 8.3125, 1.0, 1.5, 0.0, -0.08, 0.0, true),
        ];

        const POSITION_EPS: f64 = 5.0e-8;
        const VELOCITY_EPS: f64 = 5.0e-8;

        for (controller_tick, x, y, z, vx, vy, vz, on_ground) in expected {
            let row = &rows[controller_tick - 1];
            assert!(
                (row.x - x).abs() < POSITION_EPS,
                "unexpected x at controller tick {controller_tick}: {}",
                row.x
            );
            assert!(
                (row.y - y).abs() < POSITION_EPS,
                "unexpected y at controller tick {controller_tick}: {}",
                row.y
            );
            assert!(
                (row.z - z).abs() < POSITION_EPS,
                "unexpected z at controller tick {controller_tick}: {}",
                row.z
            );
            assert!(
                (row.vx - vx).abs() < VELOCITY_EPS,
                "unexpected vx at controller tick {controller_tick}: {}",
                row.vx
            );
            assert!(
                (row.vy - vy).abs() < VELOCITY_EPS,
                "unexpected vy at controller tick {controller_tick}: {}",
                row.vy
            );
            assert!(
                (row.vz - vz).abs() < VELOCITY_EPS,
                "unexpected vz at controller tick {controller_tick}: {}",
                row.vz
            );
            assert_eq!(
                row.on_ground, on_ground,
                "unexpected on_ground at controller tick {controller_tick}"
            );
        }
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
            entity_rng_seed: None,
            entity_uuid: None,
            bootstrap_fluids: false,
            entity_kind: VerifyEntityKind::Item,
            no_ai: false,

            no_gravity: false,
            fire_immune: false,
            start_fire_ticks: 0,
            item_health: None,
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
