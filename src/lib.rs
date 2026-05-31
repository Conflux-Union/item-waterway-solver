use serde::Serialize;
use std::collections::HashMap;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use time::OffsetDateTime;
use time::format_description::well_known::Rfc3339;

const WIDTH: f64 = 0.25;
const HEIGHT: f64 = 0.25;
const FLUID_MOVEMENT_THRESHOLD: f64 = 0.1;
const WATER_PUSH: f64 = 0.014;
const HORIZONTAL_WATER_DAMPING: f64 = 0.99_f32 as f64;
const HORIZONTAL_MOVEMENT_DAMPING: f64 = 0.98_f32 as f64;
const VERTICAL_MOVEMENT_DAMPING: f64 = 0.98;
const GRAVITY: f64 = 0.04;
const BUOYANCY: f64 = 5.0e-4_f32 as f64;
const BUOYANCY_CAP: f64 = 0.06_f32 as f64;
const SLIME_STEP_ON_VY_THRESHOLD: f64 = 0.1;
const SLIME_STEP_ON_BASE: f64 = 0.4;
const SLIME_STEP_ON_VY_SCALE: f64 = 0.2;
const HORIZONTAL_REST_THRESHOLD2: f64 = 1.0e-5_f32 as f64;
const AABB_DEFLATE: f64 = 0.001;
const MOVEMENT_SAMPLE_MODULO: usize = 4;
const FLUID_CURRENT_MIN_OLD_MOVEMENT: f64 = 0.003;
const FLUID_CURRENT_MIN_IMPULSE: f64 = 0.0045;
const FLUID_CURRENT_EPSILON2: f64 = 1.0e-5_f32 as f64;

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    Early,
    Full,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Args {
    pub out: PathBuf,
    pub mode: Mode,
    pub ticks: usize,
    pub top: usize,
    pub max_prefix: usize,
    pub cadence_pairs: usize,
    pub cadence_tolerance: f64,
    pub long_window: usize,
    pub start_samples: usize,
    pub keep_weak: bool,
    pub min_early_block_hit_rate: f64,
    pub early_limit: usize,
    pub long_limit: usize,
    pub dedupe_long: bool,
    pub full_cadence_pairs: usize,
    pub full_cadence_tolerance: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fixed_start_offsets: Option<Vec<f64>>,
    pub start_y: f64,
    pub start_vx: f64,
    pub start_vy: f64,
    pub start_on_ground: bool,
}

pub enum ParsedArgs {
    Help(String),
    Run(Args),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum StepOn {
    None,
    Slime,
}

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum Floor {
    Normal,
    PackedIce,
    BlueIce,
    Slime,
}

impl Floor {
    fn as_str(self) -> &'static str {
        match self {
            Self::Normal => "normal",
            Self::PackedIce => "packed_ice",
            Self::BlueIce => "blue_ice",
            Self::Slime => "slime",
        }
    }

    fn code(self) -> char {
        match self {
            Self::Normal => 'N',
            Self::PackedIce => 'I',
            Self::BlueIce => 'B',
            Self::Slime => 'S',
        }
    }

    fn friction(self) -> f64 {
        match self {
            Self::Normal => 0.6_f32 as f64,
            Self::PackedIce => 0.98_f32 as f64,
            Self::BlueIce => 0.989_f32 as f64,
            Self::Slime => 0.8_f32 as f64,
        }
    }

    fn step_on(self) -> StepOn {
        match self {
            Self::Slime => StepOn::Slime,
            _ => StepOn::None,
        }
    }
}

#[derive(Clone, Debug)]
struct Cell {
    surface: Option<f64>,
    flow: i8,
    amount: u8,
    floor: Floor,
}

impl Cell {
    fn friction(&self) -> f64 {
        self.floor.friction()
    }

    fn step_on(&self) -> StepOn {
        self.floor.step_on()
    }

    fn code(&self) -> String {
        let prefix = if self.surface.is_none() {
            'D'
        } else if self.flow < 0 {
            'R'
        } else if self.flow > 0 {
            'F'
        } else {
            'S'
        };
        format!("{}{}", prefix, self.floor.code())
    }
}

#[derive(Clone)]
struct PrefixAtom {
    name: &'static str,
    cells: Vec<Cell>,
}

#[derive(Clone)]
struct PrefixSpec {
    label: String,
    cells: Vec<Cell>,
    signature: String,
}

#[derive(Clone)]
pub struct CycleSpec {
    pub name: String,
    cells: Vec<Cell>,
    note: String,
    proven: bool,
    signature: String,
}

#[derive(Clone)]
struct Layout {
    prefix_length: usize,
    period: usize,
    total_length: usize,
    cells: Vec<Cell>,
    flow_directions: Vec<i8>,
}

#[derive(Clone)]
struct SimConfig {
    ticks: usize,
    start_x: f64,
    start_y: f64,
    start_vx: f64,
    start_vy: f64,
    entity_id_mod4: usize,
    initial_tick_count: usize,
    start_on_ground: Option<bool>,
}

#[derive(Clone)]
struct Simulation {
    xs: Vec<f64>,
    ys: Vec<f64>,
    vxs: Vec<f64>,
    vys: Vec<f64>,
    on_grounds: Vec<u8>,
    floors: Vec<Floor>,
}

#[derive(Clone)]
struct WindowMetricContext {
    vx_sum: Vec<f64>,
    vx_sq_sum: Vec<f64>,
}

#[derive(Clone, Debug)]
struct WindowMetrics {
    average_vx: f64,
    mean_vx_error: f64,
    std_vx: f64,
    average_distance_vx: f64,
    long_window_score: Option<f64>,
    long_window_start_tick: Option<usize>,
    suffix_start_tick: Option<usize>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EarlyCadenceSample {
    pub pair: usize,
    pub t0: usize,
    pub t1: usize,
    pub x0: f64,
    pub x1: f64,
    pub distance: f64,
    pub distance_error: f64,
    pub floor_delta: i32,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FullCadenceSample {
    pub pair: usize,
    pub t0: usize,
    pub t1: usize,
    pub x0: f64,
    pub x1: f64,
    pub distance: f64,
    pub distance_error: f64,
    pub floor_delta: i32,
    pub hit_margin: f64,
    pub endpoint_boundary_margin: f64,
}

#[derive(Clone, Debug)]
struct EarlyCadence {
    cadence_start_tick: usize,
    cadence_pairs: usize,
    cadence_mean_abs_distance_error: f64,
    cadence_mean_signed_distance_error: f64,
    cadence_max_abs_distance_error: f64,
    cadence_block_hit_rate: f64,
    cadence_within_tolerance_rate: f64,
    cadence_pass: bool,
    cadence_samples: Vec<EarlyCadenceSample>,
    early_cadence_score: f64,
}

#[derive(Clone, Debug)]
struct FullCadence {
    full_cadence_start_tick: usize,
    full_cadence_pairs: usize,
    full_cadence_mean_abs_distance_error: f64,
    full_cadence_mean_signed_distance_error: f64,
    full_cadence_max_abs_distance_error: f64,
    full_cadence_block_hit_rate: f64,
    full_cadence_within_tolerance_rate: f64,
    full_cadence_longest_hit_run: usize,
    full_cadence_first_miss: Option<EarlyCadenceSample>,
    full_cadence_min_hit_margin: f64,
    full_cadence_mean_hit_margin: f64,
    full_cadence_min_endpoint_boundary_margin: f64,
    full_cadence_mean_endpoint_boundary_margin: f64,
    full_cadence_samples: Vec<FullCadenceSample>,
    full_cadence_distance: f64,
    full_cadence_average_speed: f64,
}

#[derive(Clone)]
struct EarlyCandidate {
    id: String,
    early_score: f64,
    prefix_index: usize,
    cycle_index: usize,
    start_offset: f64,
    entity_id_mod4: usize,
    initial_tick_count: usize,
    cadence: EarlyCadence,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CellDescription {
    pub index: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub surface: Option<f64>,
    pub flow: i8,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub derived_flow_hint: Option<i8>,
    pub amount: u8,
    pub floor: String,
    pub code: String,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FirstTick {
    pub tick: usize,
    pub x: f64,
    pub y: f64,
    pub vx: f64,
    pub vy: f64,
    pub floor: String,
    pub on_ground: bool,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ResultRow {
    pub id: String,
    pub pass: String,
    pub score: f64,
    pub early_score: f64,
    pub prefix_label: String,
    pub prefix_length: usize,
    pub backbone: String,
    pub proven: bool,
    pub start_offset: f64,
    pub entity_id_mod4: usize,
    pub initial_tick_count: usize,
    pub period: usize,
    pub cadence_start_tick: usize,
    pub cadence_pairs: usize,
    pub cadence_mean_abs_distance_error: f64,
    pub cadence_mean_signed_distance_error: f64,
    pub cadence_max_abs_distance_error: f64,
    pub cadence_block_hit_rate: f64,
    pub cadence_within_tolerance_rate: f64,
    pub cadence_pass: bool,
    pub cadence_samples: Vec<EarlyCadenceSample>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub full_cadence_start_tick: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub full_cadence_pairs: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub full_cadence_mean_abs_distance_error: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub full_cadence_mean_signed_distance_error: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub full_cadence_max_abs_distance_error: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub full_cadence_block_hit_rate: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub full_cadence_within_tolerance_rate: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub full_cadence_longest_hit_run: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub full_cadence_average_speed: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub full_cadence_distance: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub full_cadence_first_miss: Option<EarlyCadenceSample>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub full_cadence_min_hit_margin: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub full_cadence_mean_hit_margin: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub full_cadence_min_endpoint_boundary_margin: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub full_cadence_mean_endpoint_boundary_margin: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub full_cadence_samples: Option<Vec<FullCadenceSample>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub long_window_start_tick: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub long_average_vx: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub long_mean_vx_error: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub long_std_vx: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub long_average_distance_vx: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suffix_average_vx: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suffix_mean_vx_error: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suffix_std_vx: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suffix_average_distance_vx: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub first_ticks: Option<Vec<FirstTick>>,
    pub prefix_cells: Vec<CellDescription>,
    pub cycle_cells: Vec<CellDescription>,
    pub note: String,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SearchPayload {
    pub evaluated: usize,
    pub early_kept: usize,
    pub early_deduped: usize,
    pub long_verified: usize,
    pub results: Vec<ResultRow>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ConstantsOutput {
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
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct JsonOutput {
    generated_at: String,
    args: Args,
    constants: ConstantsOutput,
    evaluated: usize,
    early_kept: usize,
    early_deduped: usize,
    long_verified: usize,
    top: Vec<ResultRow>,
}

#[derive(Clone, Copy, Debug, Default)]
struct FluidTracker {
    height: f64,
    accumulated_current_x: f64,
    current_count: usize,
}

impl FluidTracker {
    fn is_in_fluid(self) -> bool {
        self.height > 0.0
    }

    fn applies_underwater_movement(self) -> bool {
        self.height > FLUID_MOVEMENT_THRESHOLD
    }
}
impl Layout {
    fn new(prefix_cells: &[Cell], cycle_cells: &[Cell]) -> Self {
        let prefix_length = prefix_cells.len();
        let period = cycle_cells.len();
        let total_length = prefix_length + period;
        let mut cells = Vec::with_capacity(total_length);
        cells.extend_from_slice(prefix_cells);
        cells.extend_from_slice(cycle_cells);
        let flow_directions = (0..total_length)
            .map(|index| {
                compute_flow_direction(&cells, prefix_length, period, total_length, index as isize)
            })
            .collect();
        Self {
            prefix_length,
            period,
            total_length,
            cells,
            flow_directions,
        }
    }

    fn cell_index(&self, index: isize) -> Option<usize> {
        layout_cell_index(self.prefix_length, self.period, self.total_length, index)
    }

    fn cell_at(&self, index: isize) -> Option<&Cell> {
        self.cell_index(index).map(|resolved| &self.cells[resolved])
    }

    fn flow_direction_at(&self, index: isize) -> i8 {
        self.cell_index(index)
            .map(|resolved| self.flow_directions[resolved])
            .unwrap_or(0)
    }
}

pub fn usage() -> String {
    "Usage: cargo run --release -- [--out <dir>] [--ticks 500] [--top 80] [--max-prefix 8]\n\nSearches transition prefixes for a slime-piston-launched 1.17.1 item. The launch state is modeled as\nvx=1 after PistonMovingBlockEntity collides with a moving slime block. Candidates must enter a\n2gt-per-block cadence by <=5gt and keep a long-run 0.5 m/gt average.".to_string()
}

pub fn parse_args(argv: &[String]) -> Result<ParsedArgs, String> {
    let mut args = Args {
        out: PathBuf::from("artifacts").join("item-waterway-launch-search"),
        mode: Mode::Full,
        ticks: 500,
        top: 80,
        max_prefix: 8,
        cadence_pairs: 20,
        cadence_tolerance: 0.075,
        long_window: 200,
        start_samples: 33,
        keep_weak: false,
        min_early_block_hit_rate: 0.8,
        early_limit: 0,
        long_limit: 0,
        dedupe_long: true,
        full_cadence_pairs: 3000,
        full_cadence_tolerance: 0.05,
        fixed_start_offsets: None,
        start_y: 0.0,
        start_vx: 1.0,
        start_vy: 0.0,
        start_on_ground: true,
    };

    let mut i = 0;
    while i < argv.len() {
        let arg = &argv[i];
        if arg == "--help" || arg == "-h" {
            return Ok(ParsedArgs::Help(usage()));
        }
        let next = |index: &mut usize| -> Result<&str, String> {
            *index += 1;
            argv.get(*index)
                .map(|value| value.as_str())
                .ok_or_else(|| format!("Missing value for {}", arg))
        };
        match arg.as_str() {
            "--out" => args.out = PathBuf::from(next(&mut i)?),
            "--mode" => {
                args.mode = match next(&mut i)? {
                    "early" => Mode::Early,
                    "full" => Mode::Full,
                    _ => return Err("--mode must be either 'early' or 'full'.".to_string()),
                }
            }
            "--early-only" => args.mode = Mode::Early,
            "--ticks" => args.ticks = parse_usize(next(&mut i)?, "--ticks")?,
            "--top" => args.top = parse_usize(next(&mut i)?, "--top")?,
            "--max-prefix" => args.max_prefix = parse_usize(next(&mut i)?, "--max-prefix")?,
            "--cadence-pairs" => {
                args.cadence_pairs = parse_usize(next(&mut i)?, "--cadence-pairs")?
            }
            "--cadence-tolerance" => {
                args.cadence_tolerance = parse_f64(next(&mut i)?, "--cadence-tolerance")?
            }
            "--long-window" => args.long_window = parse_usize(next(&mut i)?, "--long-window")?,
            "--start-samples" => {
                args.start_samples = parse_usize(next(&mut i)?, "--start-samples")?
            }
            "--fixed-start-offsets" => {
                let values = parse_number_list(next(&mut i)?);
                args.fixed_start_offsets = Some(values);
            }
            "--start-y" => args.start_y = parse_f64(next(&mut i)?, "--start-y")?,
            "--start-vx" => args.start_vx = parse_f64(next(&mut i)?, "--start-vx")?,
            "--start-vy" => args.start_vy = parse_f64(next(&mut i)?, "--start-vy")?,
            "--start-on-ground" => {
                args.start_on_ground = match next(&mut i)?.to_ascii_lowercase().as_str() {
                    "true" => true,
                    "false" => false,
                    _ => return Err("--start-on-ground must be true or false.".to_string()),
                }
            }
            "--keep-weak" => args.keep_weak = true,
            "--min-early-block-hit-rate" => {
                args.min_early_block_hit_rate =
                    parse_f64(next(&mut i)?, "--min-early-block-hit-rate")?
            }
            "--early-limit" => args.early_limit = parse_usize(next(&mut i)?, "--early-limit")?,
            "--long-limit" => args.long_limit = parse_usize(next(&mut i)?, "--long-limit")?,
            "--no-dedupe-long" => args.dedupe_long = false,
            "--full-cadence-pairs" => {
                args.full_cadence_pairs = parse_usize(next(&mut i)?, "--full-cadence-pairs")?
            }
            "--full-cadence-tolerance" => {
                args.full_cadence_tolerance = parse_f64(next(&mut i)?, "--full-cadence-tolerance")?
            }
            _ => return Err(format!("Unknown argument: {}\n{}", arg, usage())),
        }
        i += 1;
    }

    let minimum_early_ticks = 5 + args.cadence_pairs * 2 + 4;
    if args.ticks < minimum_early_ticks {
        match args.mode {
            Mode::Early => args.ticks = minimum_early_ticks,
            Mode::Full => {
                return Err("--ticks must be at least 5 + --cadence-pairs * 2 + 4.".to_string());
            }
        }
    }
    if matches!(args.mode, Mode::Full) && args.ticks < args.long_window + 10 {
        return Err("--ticks must be at least --long-window + 10 in full mode.".to_string());
    }
    if args.max_prefix > 16 {
        return Err("--max-prefix must be in [0, 16].".to_string());
    }
    if !(2..=257).contains(&args.start_samples) {
        return Err("--start-samples must be in [2, 257].".to_string());
    }
    if let Some(values) = args.fixed_start_offsets.as_ref() {
        if values.is_empty() {
            return Err(
                "--fixed-start-offsets must contain at least one finite number.".to_string(),
            );
        }
    }
    if !args.start_y.is_finite() || !args.start_vx.is_finite() || !args.start_vy.is_finite() {
        return Err("--start-y, --start-vx, and --start-vy must be finite numbers.".to_string());
    }
    if !(0.0..=1.0).contains(&args.min_early_block_hit_rate) {
        return Err("--min-early-block-hit-rate must be in [0, 1].".to_string());
    }
    if args.full_cadence_pairs < 1 {
        return Err("--full-cadence-pairs must be >= 1.".to_string());
    }
    if !(args.full_cadence_tolerance >= 0.0 && args.full_cadence_tolerance.is_finite()) {
        return Err("--full-cadence-tolerance must be >= 0.".to_string());
    }
    if matches!(args.mode, Mode::Full) {
        args.ticks = args.ticks.max(5 + args.full_cadence_pairs * 2 + 4);
    }
    Ok(ParsedArgs::Run(args))
}

pub fn main_cli(argv: Vec<String>) -> Result<(), String> {
    let parsed = parse_args(&argv)?;
    match parsed {
        ParsedArgs::Help(text) => {
            println!("{}", text);
            Ok(())
        }
        ParsedArgs::Run(args) => {
            fs::create_dir_all(&args.out)
                .map_err(|error| format!("Failed to create output dir: {error}"))?;
            let payload = search(&args);
            let csv_path = args.out.join("launch-search-results.csv");
            let md_path = args.out.join("launch-search-summary.md");
            let json_path = args.out.join("launch-top-candidates.json");
            write_csv(&csv_path, &payload.results)?;
            write_summary(&md_path, &payload, &args)?;
            write_json(&json_path, &payload, &args)?;
            println!("Evaluated {} launch states.", payload.evaluated);
            println!("Kept {} candidates.", payload.results.len());
            println!("CSV: {}", csv_path.display());
            println!("Markdown: {}", md_path.display());
            println!("JSON: {}", json_path.display());
            println!();
            let rows = &payload.results[..payload.results.len().min(args.top.min(20))];
            println!(
                "{}",
                if matches!(args.mode, Mode::Early) {
                    markdown_early_table(rows)
                } else {
                    markdown_table(rows)
                }
            );
            Ok(())
        }
    }
}

fn parse_usize(text: &str, flag: &str) -> Result<usize, String> {
    text.parse::<usize>()
        .map_err(|_| format!("{} must be a non-negative integer.", flag))
}

fn parse_f64(text: &str, flag: &str) -> Result<f64, String> {
    text.parse::<f64>()
        .map_err(|_| format!("{} must be a finite number.", flag))
        .and_then(|value| {
            if value.is_finite() {
                Ok(value)
            } else {
                Err(format!("{} must be a finite number.", flag))
            }
        })
}

fn parse_number_list(text: &str) -> Vec<f64> {
    text.split(',')
        .filter_map(|value| value.trim().parse::<f64>().ok())
        .filter(|value| value.is_finite())
        .collect()
}

fn cell(surface: Option<f64>, flow: i8, floor: Floor, amount: Option<u8>) -> Cell {
    let fluid_amount = surface
        .map(|value| amount.unwrap_or_else(|| (value * 9.0).round() as u8))
        .unwrap_or(0);
    Cell {
        surface,
        flow,
        amount: fluid_amount,
        floor,
    }
}

fn dry_gap(length: usize, floor_pattern: &[Floor]) -> Vec<Cell> {
    let mut cells = Vec::with_capacity(length);
    for index in 0..length {
        cells.push(cell(
            None,
            0,
            floor_pattern[index % floor_pattern.len()],
            None,
        ));
    }
    cells
}

fn still_water(length: usize, floor_pattern: &[Floor], surface: f64) -> Vec<Cell> {
    let mut cells = Vec::with_capacity(length);
    for index in 0..length {
        cells.push(cell(
            Some(surface),
            0,
            floor_pattern[index % floor_pattern.len()],
            Some(8),
        ));
    }
    cells
}

fn one_way_water(
    length: usize,
    direction: i8,
    floor_pattern: &[Floor],
    full_height: bool,
) -> Vec<Cell> {
    let mut cells = Vec::with_capacity(length);
    for index in 0..length {
        let distance_from_source = if direction == 1 {
            index
        } else {
            length - 1 - index
        };
        let amount = (8_i32 - distance_from_source as i32).max(1) as u8;
        let surface = if full_height {
            1.0
        } else {
            amount as f64 / 9.0
        };
        cells.push(cell(
            Some(surface),
            direction,
            floor_pattern[index % floor_pattern.len()],
            Some(amount),
        ));
    }
    cells
}

fn cycle_definition(
    name: impl Into<String>,
    cells: Vec<Cell>,
    note: impl Into<String>,
    proven: bool,
) -> CycleSpec {
    let cells_signature = cells_signature(&cells);
    CycleSpec {
        name: name.into(),
        cells,
        note: note.into(),
        proven,
        signature: cells_signature,
    }
}

pub fn backbone_cycles() -> Vec<CycleSpec> {
    let mut cycles = vec![
        cycle_definition(
            "W3-I_D3-B",
            [
                one_way_water(3, 1, &[Floor::PackedIce], false),
                dry_gap(3, &[Floor::BlueIce]),
            ]
            .concat(),
            "Proven long-run backbone: 3 forward water cells on packed ice, 3 dry cells on blue ice.",
            true,
        ),
        cycle_definition(
            "W2-I_D2-B",
            [
                one_way_water(2, 1, &[Floor::PackedIce], false),
                dry_gap(2, &[Floor::BlueIce]),
            ]
            .concat(),
            "Dry-gap compact variant from source model search; needs game verification.",
            false,
        ),
        cycle_definition(
            "W2-I_S2-I",
            [
                one_way_water(2, 1, &[Floor::PackedIce], false),
                still_water(2, &[Floor::PackedIce], 8.0 / 9.0),
            ]
            .concat(),
            "Still-source compact variant; build with intentional source-water cells only.",
            false,
        ),
        cycle_definition(
            "W2-B_S2-B",
            [
                one_way_water(2, 1, &[Floor::BlueIce], false),
                still_water(2, &[Floor::BlueIce], 8.0 / 9.0),
            ]
            .concat(),
            "Still-source compact blue-ice variant; build with intentional source-water cells only.",
            false,
        ),
        cycle_definition(
            "W2-I_R2-I_D1-B",
            [
                one_way_water(2, 1, &[Floor::PackedIce], false),
                one_way_water(2, -1, &[Floor::PackedIce], false),
                dry_gap(1, &[Floor::BlueIce]),
            ]
            .concat(),
            "Reverse-water compact variant with a real two-cell water gradient; needs game verification.",
            false,
        ),
    ];

    let water_floor_sets: [(&str, &[Floor]); 4] = [
        ("I", &[Floor::PackedIce]),
        ("B", &[Floor::BlueIce]),
        ("IB", &[Floor::PackedIce, Floor::BlueIce]),
        ("BI", &[Floor::BlueIce, Floor::PackedIce]),
    ];
    let gap_floor_sets = water_floor_sets;

    for (water_name, water_floors) in water_floor_sets {
        for (gap_name, gap_floors) in gap_floor_sets {
            cycles.push(cycle_definition(
                format!("W2-{}_D2-{}", water_name, gap_name),
                [
                    one_way_water(2, 1, water_floors, false),
                    dry_gap(2, gap_floors),
                ]
                .concat(),
                "Generated compact dry-gap variant; needs game verification.",
                false,
            ));
            cycles.push(cycle_definition(
                format!("W2-{}_S2-{}", water_name, gap_name),
                [one_way_water(2, 1, water_floors, false), still_water(2, gap_floors, 8.0 / 9.0)].concat(),
                "Generated compact source/still-water variant; build with intentional source-water cells only.",
                false,
            ));
        }
    }

    cycles
}

fn prefix_atoms() -> Vec<PrefixAtom> {
    vec![
        PrefixAtom {
            name: "DN",
            cells: dry_gap(1, &[Floor::Normal]),
        },
        PrefixAtom {
            name: "DI",
            cells: dry_gap(1, &[Floor::PackedIce]),
        },
        PrefixAtom {
            name: "DB",
            cells: dry_gap(1, &[Floor::BlueIce]),
        },
        PrefixAtom {
            name: "DS",
            cells: dry_gap(1, &[Floor::Slime]),
        },
        PrefixAtom {
            name: "SN",
            cells: still_water(1, &[Floor::Normal], 8.0 / 9.0),
        },
        PrefixAtom {
            name: "SI",
            cells: still_water(1, &[Floor::PackedIce], 8.0 / 9.0),
        },
        PrefixAtom {
            name: "SB",
            cells: still_water(1, &[Floor::BlueIce], 8.0 / 9.0),
        },
        PrefixAtom {
            name: "R2N",
            cells: one_way_water(2, -1, &[Floor::Normal], false),
        },
        PrefixAtom {
            name: "R2I",
            cells: one_way_water(2, -1, &[Floor::PackedIce], false),
        },
        PrefixAtom {
            name: "R2B",
            cells: one_way_water(2, -1, &[Floor::BlueIce], false),
        },
        PrefixAtom {
            name: "R3N",
            cells: one_way_water(3, -1, &[Floor::Normal], false),
        },
        PrefixAtom {
            name: "R3I",
            cells: one_way_water(3, -1, &[Floor::PackedIce], false),
        },
        PrefixAtom {
            name: "R3B",
            cells: one_way_water(3, -1, &[Floor::BlueIce], false),
        },
        PrefixAtom {
            name: "F2N",
            cells: one_way_water(2, 1, &[Floor::Normal], false),
        },
        PrefixAtom {
            name: "F2I",
            cells: one_way_water(2, 1, &[Floor::PackedIce], false),
        },
        PrefixAtom {
            name: "F2B",
            cells: one_way_water(2, 1, &[Floor::BlueIce], false),
        },
    ]
}

fn generate_prefixes(max_cells: usize, atoms: &[PrefixAtom]) -> Vec<PrefixSpec> {
    let mut prefixes = Vec::new();
    prefixes.push(PrefixSpec {
        label: "none".to_string(),
        cells: Vec::new(),
        signature: String::new(),
    });

    let mut stack: Vec<(Vec<usize>, usize)> = vec![(Vec::new(), 0)];
    while let Some((current_indices, current_length)) = stack.pop() {
        for (atom_index, atom) in atoms.iter().enumerate() {
            let next_length = current_length + atom.cells.len();
            if next_length > max_cells {
                continue;
            }
            let mut next_indices = current_indices.clone();
            next_indices.push(atom_index);
            let spec = prefix_spec_from_indices(&next_indices, atoms);
            prefixes.push(spec);
            stack.push((next_indices, next_length));
        }
    }
    prefixes
}

fn prefix_spec_from_indices(indices: &[usize], atoms: &[PrefixAtom]) -> PrefixSpec {
    let mut label = String::new();
    let total_len = indices.iter().map(|&index| atoms[index].cells.len()).sum();
    let mut cells = Vec::with_capacity(total_len);
    for (offset, &index) in indices.iter().enumerate() {
        if offset > 0 {
            label.push('-');
        }
        label.push_str(atoms[index].name);
        cells.extend_from_slice(&atoms[index].cells);
    }
    PrefixSpec {
        label,
        signature: cells_signature(&cells),
        cells,
    }
}

fn cells_signature(cells: &[Cell]) -> String {
    let mut out = String::new();
    for (index, cell) in cells.iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        let _ = write!(
            out,
            "{}:{}:{}:{}",
            cell.amount,
            cell.floor.as_str(),
            match cell.step_on() {
                StepOn::None => "none",
                StepOn::Slime => "slime",
            },
            cell.friction()
        );
    }
    out
}

fn layout_cell_index(
    prefix_length: usize,
    period: usize,
    total_length: usize,
    index: isize,
) -> Option<usize> {
    if total_length == 0 {
        return None;
    }
    if index < prefix_length as isize {
        return (index >= 0).then_some(index as usize);
    }
    if period == 0 {
        return None;
    }
    let cycle_offset = (index - prefix_length as isize).rem_euclid(period as isize) as usize;
    Some(prefix_length + cycle_offset)
}

fn compute_flow_direction(
    cells: &[Cell],
    prefix_length: usize,
    period: usize,
    total_length: usize,
    index: isize,
) -> i8 {
    let Some(current_index) = layout_cell_index(prefix_length, period, total_length, index) else {
        return 0;
    };
    let current = &cells[current_index];
    if current.amount == 0 {
        return 0;
    }

    let own_height = current.amount as f64 / 9.0;
    let mut horizontal = 0.0;
    for (neighbor_index, step_x) in [(index - 1, -1.0), (index + 1, 1.0)] {
        if let Some(resolved) =
            layout_cell_index(prefix_length, period, total_length, neighbor_index)
        {
            let neighbor = &cells[resolved];
            if neighbor.amount > 0 {
                horizontal += step_x * (own_height - neighbor.amount as f64 / 9.0);
            }
        }
    }

    if horizontal > 1.0e-12 {
        1
    } else if horizontal < -1.0e-12 {
        -1
    } else {
        0
    }
}

fn update_fluid_tracker(layout: &Layout, x: f64, y: f64, ignore_current: bool) -> FluidTracker {
    let half_width = WIDTH / 2.0;
    let box_min_x = x - half_width + AABB_DEFLATE;
    let box_max_x = x + half_width - AABB_DEFLATE;
    let box_min_y = y + AABB_DEFLATE;
    let box_max_y = y + HEIGHT - AABB_DEFLATE;
    let x0 = box_min_x.floor() as isize;
    let x1 = box_max_x.ceil() as isize - 1;
    let y0 = box_min_y.floor() as isize;
    let y1 = box_max_y.ceil() as isize - 1;
    let mut tracker = FluidTracker::default();

    for ix in x0..=x1 {
        let Some(current) = layout.cell_at(ix) else {
            continue;
        };
        let Some(surface) = current.surface else {
            continue;
        };
        for iy in y0..=y1 {
            if iy != 0 {
                continue;
            }
            let fluid_bottom = iy as f64;
            let fluid_top = fluid_bottom + surface;
            if fluid_top < box_min_y {
                continue;
            }
            tracker.height = tracker.height.max(fluid_top - y);
            if !ignore_current {
                let mut flow_x = layout.flow_direction_at(ix) as f64;
                if tracker.height < 0.4 {
                    flow_x *= tracker.height;
                }
                tracker.accumulated_current_x += flow_x;
                tracker.current_count += 1;
            }
        }
    }

    tracker
}

fn floor_at(layout: &Layout, x: f64) -> Floor {
    layout
        .cell_at(x.floor() as isize)
        .map(|cell| cell.floor)
        .unwrap_or(Floor::Normal)
}

fn apply_fluid_current(vx: f64, tracker: FluidTracker) -> f64 {
    if tracker.current_count == 0
        || tracker.accumulated_current_x * tracker.accumulated_current_x < FLUID_CURRENT_EPSILON2
    {
        return vx;
    }

    let direction = tracker.accumulated_current_x.signum();
    if direction == 0.0 {
        return vx;
    }

    let mut impulse = direction * WATER_PUSH;
    if vx.abs() < FLUID_CURRENT_MIN_OLD_MOVEMENT && impulse.abs() < FLUID_CURRENT_MIN_IMPULSE {
        impulse = direction * FLUID_CURRENT_MIN_IMPULSE;
    }
    vx + impulse
}
fn horizontal_drag(floor: Floor, on_ground: bool, vy: f64) -> f64 {
    let mut drag = HORIZONTAL_MOVEMENT_DAMPING;
    if on_ground {
        drag = floor.friction() * HORIZONTAL_MOVEMENT_DAMPING;
        if matches!(floor.step_on(), StepOn::Slime) && vy.abs() < SLIME_STEP_ON_VY_THRESHOLD {
            drag *= SLIME_STEP_ON_BASE + SLIME_STEP_ON_VY_SCALE * vy.abs();
        }
    }
    drag
}

fn vertical_velocity_after_landing(floor: Floor, vy: f64) -> f64 {
    if matches!(floor.step_on(), StepOn::Slime) {
        if vy < 0.0 { -vy * 0.8 } else { vy }
    } else {
        0.0
    }
}

fn simulate(layout: &Layout, config: &SimConfig) -> Simulation {
    let n = config.ticks + 1;
    let mut xs = vec![0.0; n];
    let mut ys = vec![0.0; n];
    let mut vxs = vec![0.0; n];
    let mut vys = vec![0.0; n];
    let mut on_grounds = vec![0_u8; n];
    let mut floors = vec![Floor::Normal; n];

    let mut x = config.start_x;
    let mut y = config.start_y;
    let mut vx = config.start_vx;
    let mut vy = config.start_vy;
    let mut was_on_ground = config.start_on_ground.unwrap_or(config.start_y <= 0.0);
    let mut fluid_tracker = update_fluid_tracker(layout, x, y, true);

    xs[0] = x;
    ys[0] = y;
    vxs[0] = vx;
    vys[0] = vy;
    on_grounds[0] = u8::from(was_on_ground);
    floors[0] = floor_at(layout, x);

    for tick in 1..=config.ticks {
        let tick_count = config.initial_tick_count + tick;
        if fluid_tracker.is_in_fluid() && fluid_tracker.applies_underwater_movement() {
            vx *= HORIZONTAL_WATER_DAMPING;
            if vy < BUOYANCY_CAP {
                vy += BUOYANCY;
            }
        } else {
            vy -= GRAVITY;
        }

        let phase_mod4 = (tick_count + config.entity_id_mod4) % MOVEMENT_SAMPLE_MODULO;
        let should_move = !was_on_ground || vx * vx > HORIZONTAL_REST_THRESHOLD2 || phase_mod4 == 0;
        let mut on_ground = was_on_ground;

        if should_move {
            x += vx;
            y += vy;

            on_ground = false;
            if y < 0.0 {
                on_ground = vy < 0.0;
                y = 0.0;
            }

            let floor = floor_at(layout, x);
            if on_ground {
                vy = vertical_velocity_after_landing(floor, vy);
            }
            vx *= horizontal_drag(floor, on_ground, vy);
            vy *= VERTICAL_MOVEMENT_DAMPING;
            if on_ground && vy < 0.0 {
                vy *= -0.5;
            }
        }

        fluid_tracker = update_fluid_tracker(layout, x, y, false);
        if fluid_tracker.is_in_fluid() {
            vx = apply_fluid_current(vx, fluid_tracker);
        }

        xs[tick] = x;
        ys[tick] = y;
        vxs[tick] = vx;
        vys[tick] = vy;
        on_grounds[tick] = u8::from(on_ground);
        floors[tick] = floor_at(layout, x);
        was_on_ground = on_ground;
    }

    Simulation {
        xs,
        ys,
        vxs,
        vys,
        on_grounds,
        floors,
    }
}

fn window_metric_context(sim: &Simulation) -> WindowMetricContext {
    let mut vx_sum = vec![0.0; sim.vxs.len() + 1];
    let mut vx_sq_sum = vec![0.0; sim.vxs.len() + 1];

    for index in 0..sim.vxs.len() {
        vx_sum[index + 1] = vx_sum[index] + sim.vxs[index];
        vx_sq_sum[index + 1] = vx_sq_sum[index] + sim.vxs[index] * sim.vxs[index];
    }

    WindowMetricContext { vx_sum, vx_sq_sum }
}

fn range_sum(prefix: &[f64], start_inclusive: usize, end_exclusive: usize) -> f64 {
    prefix[end_exclusive] - prefix[start_inclusive]
}

fn window_metrics(
    sim: &Simulation,
    start_tick: usize,
    window_length: usize,
    context: &WindowMetricContext,
) -> Option<WindowMetrics> {
    let state_start = start_tick + 1;
    let state_end = start_tick + window_length + 1;
    if state_end > sim.xs.len() {
        return None;
    }

    let count = window_length as f64;
    let avg_vx = range_sum(&context.vx_sum, state_start, state_end) / count;
    let vx_var =
        (range_sum(&context.vx_sq_sum, state_start, state_end) / count - avg_vx * avg_vx).max(0.0);
    Some(WindowMetrics {
        average_vx: avg_vx,
        mean_vx_error: (avg_vx - 0.5).abs(),
        std_vx: vx_var.sqrt(),
        average_distance_vx: (sim.xs[start_tick + window_length] - sim.xs[start_tick]) / count,
        long_window_score: None,
        long_window_start_tick: None,
        suffix_start_tick: None,
    })
}

fn cadence_metrics(
    sim: &Simulation,
    start_tick: usize,
    pair_count: usize,
    tolerance: f64,
) -> Option<EarlyCadence> {
    let mut max_abs_distance_error: f64 = 0.0;
    let mut mean_abs_distance_error = 0.0;
    let mut mean_signed_distance_error = 0.0;
    let mut block_hits = 0_usize;
    let mut within_tol = 0_usize;
    let mut samples = Vec::with_capacity(pair_count.min(12));

    for pair in 0..pair_count {
        let t0 = start_tick + pair * 2;
        let t1 = t0 + 2;
        if t1 >= sim.xs.len() {
            return None;
        }
        let distance = sim.xs[t1] - sim.xs[t0];
        let distance_error = distance - 1.0;
        let abs_error = distance_error.abs();
        let floor_delta = sim.xs[t1].floor() as i32 - sim.xs[t0].floor() as i32;
        max_abs_distance_error = max_abs_distance_error.max(abs_error);
        mean_abs_distance_error += abs_error;
        mean_signed_distance_error += distance_error;
        if floor_delta == 1 {
            block_hits += 1;
        }
        if abs_error <= tolerance {
            within_tol += 1;
        }
        if pair < 12 {
            samples.push(EarlyCadenceSample {
                pair,
                t0,
                t1,
                x0: sim.xs[t0],
                x1: sim.xs[t1],
                distance,
                distance_error,
                floor_delta,
            });
        }
    }

    Some(EarlyCadence {
        cadence_start_tick: start_tick,
        cadence_pairs: pair_count,
        cadence_mean_abs_distance_error: mean_abs_distance_error / pair_count as f64,
        cadence_mean_signed_distance_error: mean_signed_distance_error / pair_count as f64,
        cadence_max_abs_distance_error: max_abs_distance_error,
        cadence_block_hit_rate: block_hits as f64 / pair_count as f64,
        cadence_within_tolerance_rate: within_tol as f64 / pair_count as f64,
        cadence_pass: within_tol == pair_count && block_hits == pair_count,
        cadence_samples: samples,
        early_cadence_score: 0.0,
    })
}

fn distance_to_integer_boundary(value: f64) -> f64 {
    let fraction = value - value.floor();
    fraction.min(1.0 - fraction)
}

fn full_cadence_metrics(
    sim: &Simulation,
    start_tick: usize,
    pair_count: usize,
    tolerance: f64,
) -> Option<FullCadence> {
    let mut max_abs_distance_error: f64 = 0.0;
    let mut mean_abs_distance_error = 0.0;
    let mut mean_signed_distance_error = 0.0;
    let mut block_hits = 0_usize;
    let mut within_tol = 0_usize;
    let mut first_miss = None;
    let mut longest_hit_run = 0_usize;
    let mut current_hit_run = 0_usize;
    let mut min_hit_margin = f64::INFINITY;
    let mut mean_hit_margin = 0.0;
    let mut min_endpoint_boundary_margin = f64::INFINITY;
    let mut mean_endpoint_boundary_margin = 0.0;
    let mut samples = Vec::with_capacity(24);

    for pair in 0..pair_count {
        let t0 = start_tick + pair * 2;
        let t1 = t0 + 2;
        if t1 >= sim.xs.len() {
            return None;
        }

        let distance = sim.xs[t1] - sim.xs[t0];
        let distance_error = distance - 1.0;
        let abs_error = distance_error.abs();
        let floor0 = sim.xs[t0].floor() as i32;
        let floor_delta = sim.xs[t1].floor() as i32 - floor0;
        let hit_margin = (sim.xs[t1] - (floor0 + 1) as f64).min((floor0 + 2) as f64 - sim.xs[t1]);
        let endpoint_boundary_margin =
            distance_to_integer_boundary(sim.xs[t0]).min(distance_to_integer_boundary(sim.xs[t1]));
        let hit = floor_delta == 1;

        max_abs_distance_error = max_abs_distance_error.max(abs_error);
        mean_abs_distance_error += abs_error;
        mean_signed_distance_error += distance_error;
        min_hit_margin = min_hit_margin.min(hit_margin);
        mean_hit_margin += hit_margin;
        min_endpoint_boundary_margin = min_endpoint_boundary_margin.min(endpoint_boundary_margin);
        mean_endpoint_boundary_margin += endpoint_boundary_margin;

        if hit {
            block_hits += 1;
            current_hit_run += 1;
            longest_hit_run = longest_hit_run.max(current_hit_run);
        } else {
            current_hit_run = 0;
            if first_miss.is_none() {
                first_miss = Some(EarlyCadenceSample {
                    pair,
                    t0,
                    t1,
                    x0: sim.xs[t0],
                    x1: sim.xs[t1],
                    distance,
                    distance_error,
                    floor_delta,
                });
            }
        }

        if abs_error <= tolerance {
            within_tol += 1;
        }
        if pair < 12 || (!hit && samples.len() < 24) {
            samples.push(FullCadenceSample {
                pair,
                t0,
                t1,
                x0: sim.xs[t0],
                x1: sim.xs[t1],
                distance,
                distance_error,
                floor_delta,
                hit_margin,
                endpoint_boundary_margin,
            });
        }
    }

    Some(FullCadence {
        full_cadence_start_tick: start_tick,
        full_cadence_pairs: pair_count,
        full_cadence_mean_abs_distance_error: mean_abs_distance_error / pair_count as f64,
        full_cadence_mean_signed_distance_error: mean_signed_distance_error / pair_count as f64,
        full_cadence_max_abs_distance_error: max_abs_distance_error,
        full_cadence_block_hit_rate: block_hits as f64 / pair_count as f64,
        full_cadence_within_tolerance_rate: within_tol as f64 / pair_count as f64,
        full_cadence_longest_hit_run: longest_hit_run,
        full_cadence_first_miss: first_miss,
        full_cadence_min_hit_margin: min_hit_margin,
        full_cadence_mean_hit_margin: mean_hit_margin / pair_count as f64,
        full_cadence_min_endpoint_boundary_margin: min_endpoint_boundary_margin,
        full_cadence_mean_endpoint_boundary_margin: mean_endpoint_boundary_margin
            / pair_count as f64,
        full_cadence_samples: samples,
        full_cadence_distance: sim.xs[start_tick + pair_count * 2] - sim.xs[start_tick],
        full_cadence_average_speed: (sim.xs[start_tick + pair_count * 2] - sim.xs[start_tick])
            / (pair_count * 2) as f64,
    })
}

fn best_early_cadence(
    sim: &Simulation,
    max_start_tick: usize,
    pair_count: usize,
    tolerance: f64,
) -> Option<EarlyCadence> {
    let mut best = None;
    let mut best_score = f64::INFINITY;
    for start_tick in 0..=max_start_tick {
        let Some(mut metrics) = cadence_metrics(sim, start_tick, pair_count, tolerance) else {
            continue;
        };
        let score = metrics.cadence_mean_abs_distance_error * 1000.0
            + metrics.cadence_max_abs_distance_error * 100.0
            + (1.0 - metrics.cadence_block_hit_rate) * 500.0
            + start_tick as f64 * 2.0;
        if score < best_score {
            best_score = score;
            metrics.early_cadence_score = score;
            best = Some(metrics);
        }
    }
    best
}

fn best_long_window(sim: &Simulation, window_length: usize) -> Option<WindowMetrics> {
    let mut best = None;
    let mut best_score = f64::INFINITY;
    let context = window_metric_context(sim);
    for start_tick in 0..sim.xs.len() {
        let Some(mut metrics) = window_metrics(sim, start_tick, window_length, &context) else {
            continue;
        };
        let score = metrics.mean_vx_error * 1000.0
            + metrics.std_vx * 20.0
            + (metrics.average_distance_vx - 0.5).abs() * 1000.0;
        if score < best_score {
            best_score = score;
            metrics.long_window_score = Some(score);
            metrics.long_window_start_tick = Some(start_tick);
            best = Some(metrics);
        }
    }
    best
}

fn suffix_long_metrics(
    sim: &Simulation,
    start_tick: usize,
    window_length: usize,
) -> Option<WindowMetrics> {
    let safe_start = start_tick.max(5);
    if safe_start + window_length >= sim.xs.len() {
        return None;
    }
    let context = window_metric_context(sim);
    let mut metrics = window_metrics(sim, safe_start, window_length, &context)?;
    metrics.suffix_start_tick = Some(safe_start);
    Some(metrics)
}

fn early_candidate_score(early: &EarlyCadence, prefix_length: usize) -> f64 {
    early.cadence_mean_abs_distance_error * 1000.0
        + early.cadence_max_abs_distance_error * 250.0
        + (1.0 - early.cadence_block_hit_rate) * 1000.0
        + (1.0 - early.cadence_within_tolerance_rate) * 500.0
        + ((early.cadence_start_tick as i32 - 2).max(0) as f64) * 10.0
        + prefix_length as f64 * 0.5
}

fn candidate_score(
    early: &EarlyCadence,
    long: &WindowMetrics,
    suffix: Option<&WindowMetrics>,
    prefix_length: usize,
    prefix_label: &str,
    proven: bool,
) -> f64 {
    let early_penalty = if early.cadence_pass { 0.0 } else { 10_000.0 };
    let long_penalty = long.mean_vx_error * 5000.0
        + long.std_vx * 100.0
        + (long.average_distance_vx - 0.5).abs() * 5000.0;
    let suffix_penalty = suffix
        .map(|value| {
            value.mean_vx_error * 3000.0
                + value.std_vx * 50.0
                + (value.average_distance_vx - 0.5).abs() * 3000.0
        })
        .unwrap_or(1000.0);
    let prefix_penalty = prefix_length as f64 * 2.0;
    let source_penalty = if prefix_label.contains('S') { 3.0 } else { 0.0 };
    let unproven_penalty = if proven { 0.0 } else { 20.0 };
    early_penalty
        + early.early_cadence_score
        + long_penalty
        + suffix_penalty
        + prefix_penalty
        + source_penalty
        + unproven_penalty
}

fn verified_candidate_score(
    early: &EarlyCadence,
    full: &FullCadence,
    long: &WindowMetrics,
    suffix: Option<&WindowMetrics>,
    prefix_length: usize,
    prefix_label: &str,
    proven: bool,
) -> f64 {
    let miss_penalty = (1.0 - full.full_cadence_block_hit_rate) * 100_000.0;
    let full_distance_penalty = (full.full_cadence_average_speed - 0.5).abs() * 20_000.0
        + full.full_cadence_mean_abs_distance_error * 1000.0
        + full.full_cadence_max_abs_distance_error * 100.0;
    candidate_score(early, long, suffix, prefix_length, prefix_label, proven)
        + miss_penalty
        + full_distance_penalty
}

fn classify_candidate(
    early: &EarlyCadence,
    long: &WindowMetrics,
    suffix: Option<&WindowMetrics>,
) -> &'static str {
    let early_strong = early.cadence_pass
        && early.cadence_start_tick <= 5
        && early.cadence_mean_abs_distance_error <= 0.025
        && early.cadence_max_abs_distance_error <= 0.075;
    let long_strong = long.mean_vx_error <= 0.005
        && (long.average_distance_vx - 0.5).abs() <= 0.005
        && long.std_vx <= 0.04;
    let suffix_ok = suffix
        .map(|value| value.mean_vx_error <= 0.02 && (value.average_distance_vx - 0.5).abs() <= 0.02)
        .unwrap_or(false);
    if early_strong && long_strong && suffix_ok {
        return "strong";
    }
    if early.cadence_start_tick <= 5
        && early.cadence_block_hit_rate >= 0.95
        && long.mean_vx_error <= 0.01
    {
        return "usable";
    }
    "weak"
}

fn classify_verified_candidate(
    early: &EarlyCadence,
    full: &FullCadence,
    long: &WindowMetrics,
    suffix: Option<&WindowMetrics>,
) -> &'static str {
    let base = classify_candidate(early, long, suffix);
    if base == "strong"
        && full.full_cadence_block_hit_rate == 1.0
        && full.full_cadence_within_tolerance_rate >= 0.98
        && (full.full_cadence_average_speed - 0.5).abs() <= 0.001
    {
        return "strong";
    }
    if early.cadence_start_tick <= 5
        && full.full_cadence_block_hit_rate >= 0.999
        && full.full_cadence_within_tolerance_rate >= 0.95
        && (full.full_cadence_average_speed - 0.5).abs() <= 0.0025
    {
        return "usable";
    }
    "weak"
}

fn stable_id(text: &str) -> String {
    let mut hash = 2_166_136_261_u32;
    for byte in text.bytes() {
        hash ^= byte as u32;
        hash = hash.wrapping_mul(16_777_619);
    }
    format!("{:08x}", hash)
}

fn candidate_dedupe_key(
    candidate: &EarlyCandidate,
    prefixes: &[PrefixSpec],
    cycles: &[CycleSpec],
) -> String {
    format!(
        "{}|{}||{:.9}||{}",
        prefixes[candidate.prefix_index].signature,
        cycles[candidate.cycle_index].signature,
        candidate.start_offset,
        candidate.initial_tick_count
    )
}

fn dedupe_early_candidates(
    candidates: Vec<EarlyCandidate>,
    prefixes: &[PrefixSpec],
    cycles: &[CycleSpec],
) -> Vec<EarlyCandidate> {
    let mut ordered: Vec<EarlyCandidate> = Vec::new();
    let mut positions: HashMap<String, usize> = HashMap::new();
    for candidate in candidates {
        let key = candidate_dedupe_key(&candidate, prefixes, cycles);
        if let Some(&position) = positions.get(&key) {
            if candidate.early_score < ordered[position].early_score {
                ordered[position] = candidate;
            }
            continue;
        }
        positions.insert(key, ordered.len());
        ordered.push(candidate);
    }
    ordered.sort_by(|left, right| left.early_score.total_cmp(&right.early_score));
    ordered
}

pub fn search(args: &Args) -> SearchPayload {
    let cycles = backbone_cycles();
    let atoms = prefix_atoms();
    let prefixes = generate_prefixes(args.max_prefix, &atoms);
    let start_offsets = if let Some(offsets) = args.fixed_start_offsets.clone() {
        offsets
    } else {
        (0..args.start_samples)
            .map(|index| 0.125 + index as f64 * (0.75 / (args.start_samples - 1) as f64))
            .collect::<Vec<_>>()
    };
    let early_ticks = args.ticks.min(5 + args.cadence_pairs * 2 + 4);

    let mut evaluated = 0_usize;
    let mut early_candidates = Vec::new();

    for (cycle_index, cycle) in cycles.iter().enumerate() {
        for (prefix_index, prefix) in prefixes.iter().enumerate() {
            let layout = Layout::new(&prefix.cells, &cycle.cells);
            for &start_offset in &start_offsets {
                for entity_id_mod4 in 0..4 {
                    let config = SimConfig {
                        ticks: early_ticks,
                        start_x: start_offset,
                        start_y: args.start_y,
                        start_vx: args.start_vx,
                        start_vy: args.start_vy,
                        entity_id_mod4,
                        initial_tick_count: 0,
                        start_on_ground: Some(args.start_on_ground),
                    };
                    evaluated += 1;
                    let early_sim = simulate(&layout, &config);
                    let Some(early) = best_early_cadence(
                        &early_sim,
                        5,
                        args.cadence_pairs,
                        args.cadence_tolerance,
                    ) else {
                        continue;
                    };
                    if !early.cadence_pass
                        && (!args.keep_weak
                            || early.cadence_block_hit_rate < args.min_early_block_hit_rate)
                    {
                        continue;
                    }
                    let id_text = format!(
                        "{}|{}|start={}|id={}|tick=0",
                        prefix.label, cycle.name, start_offset, entity_id_mod4
                    );
                    early_candidates.push(EarlyCandidate {
                        id: stable_id(&id_text),
                        early_score: early_candidate_score(&early, prefix.cells.len()),
                        prefix_index,
                        cycle_index,
                        start_offset,
                        entity_id_mod4,
                        initial_tick_count: 0,
                        cadence: early,
                    });
                }
            }
        }
    }

    early_candidates.sort_by(|left, right| left.early_score.total_cmp(&right.early_score));

    let ranked_early_candidates = if args.dedupe_long {
        dedupe_early_candidates(early_candidates.clone(), &prefixes, &cycles)
    } else {
        early_candidates.clone()
    };

    if matches!(args.mode, Mode::Early) {
        let rows = ranked_early_candidates
            .iter()
            .take(if args.early_limit > 0 {
                args.early_limit
            } else {
                ranked_early_candidates.len()
            })
            .map(|candidate| {
                early_result_row(
                    candidate,
                    &prefixes[candidate.prefix_index],
                    &cycles[candidate.cycle_index],
                )
            })
            .collect();
        return SearchPayload {
            evaluated,
            early_kept: early_candidates.len(),
            early_deduped: ranked_early_candidates.len(),
            long_verified: 0,
            results: rows,
        };
    }

    let early_limited: Vec<_> = if args.early_limit > 0 {
        ranked_early_candidates
            .iter()
            .take(args.early_limit)
            .cloned()
            .collect()
    } else {
        ranked_early_candidates.clone()
    };
    let candidates_to_verify: Vec<_> = if args.long_limit > 0 {
        early_limited
            .iter()
            .take(args.long_limit)
            .cloned()
            .collect()
    } else {
        early_limited
    };

    let mut results = Vec::new();
    for candidate in &candidates_to_verify {
        let prefix = &prefixes[candidate.prefix_index];
        let cycle = &cycles[candidate.cycle_index];
        let layout = Layout::new(&prefix.cells, &cycle.cells);
        let config = SimConfig {
            ticks: args.ticks,
            start_x: candidate.start_offset,
            start_y: args.start_y,
            start_vx: args.start_vx,
            start_vy: args.start_vy,
            entity_id_mod4: candidate.entity_id_mod4,
            initial_tick_count: candidate.initial_tick_count,
            start_on_ground: Some(args.start_on_ground),
        };
        let sim = simulate(&layout, &config);
        let Some(long) = best_long_window(&sim, args.long_window) else {
            continue;
        };
        let suffix = suffix_long_metrics(
            &sim,
            candidate.cadence.cadence_start_tick,
            args.long_window.min(args.ticks.saturating_sub(10)),
        );
        let Some(full) = full_cadence_metrics(
            &sim,
            candidate.cadence.cadence_start_tick,
            args.full_cadence_pairs,
            args.full_cadence_tolerance,
        ) else {
            continue;
        };
        let pass = classify_verified_candidate(&candidate.cadence, &full, &long, suffix.as_ref());
        if pass == "weak" && !candidate.cadence.cadence_pass && !args.keep_weak {
            continue;
        }
        let score = verified_candidate_score(
            &candidate.cadence,
            &full,
            &long,
            suffix.as_ref(),
            prefix.cells.len(),
            &prefix.label,
            cycle.proven,
        );
        results.push(full_result_row(
            candidate,
            prefix,
            cycle,
            pass,
            score,
            &full,
            &long,
            suffix.as_ref(),
            &sim,
        ));
    }

    results.sort_by(|left, right| left.score.total_cmp(&right.score));

    SearchPayload {
        evaluated,
        early_kept: early_candidates.len(),
        early_deduped: ranked_early_candidates.len(),
        long_verified: candidates_to_verify.len(),
        results,
    }
}

fn layout_cells_description(cells: &[Cell]) -> Vec<CellDescription> {
    cells
        .iter()
        .enumerate()
        .map(|(index, cell)| CellDescription {
            index,
            surface: cell.surface,
            flow: cell.flow,
            derived_flow_hint: (cell.amount == 0).then_some(0),
            amount: cell.amount,
            floor: cell.floor.as_str().to_string(),
            code: cell.code(),
        })
        .collect()
}

fn first_ticks(sim: &Simulation) -> Vec<FirstTick> {
    (0..sim.xs.len().min(16))
        .map(|tick| FirstTick {
            tick,
            x: sim.xs[tick],
            y: sim.ys[tick],
            vx: sim.vxs[tick],
            vy: sim.vys[tick],
            floor: sim.floors[tick].as_str().to_string(),
            on_ground: sim.on_grounds[tick] != 0,
        })
        .collect()
}

fn early_result_row(
    candidate: &EarlyCandidate,
    prefix: &PrefixSpec,
    cycle: &CycleSpec,
) -> ResultRow {
    ResultRow {
        id: candidate.id.clone(),
        pass: if candidate.cadence.cadence_pass {
            "early".to_string()
        } else {
            "weak-early".to_string()
        },
        score: candidate.early_score,
        early_score: candidate.early_score,
        prefix_label: prefix.label.clone(),
        prefix_length: prefix.cells.len(),
        backbone: cycle.name.clone(),
        proven: cycle.proven,
        start_offset: candidate.start_offset,
        entity_id_mod4: candidate.entity_id_mod4,
        initial_tick_count: candidate.initial_tick_count,
        period: cycle.cells.len(),
        cadence_start_tick: candidate.cadence.cadence_start_tick,
        cadence_pairs: candidate.cadence.cadence_pairs,
        cadence_mean_abs_distance_error: candidate.cadence.cadence_mean_abs_distance_error,
        cadence_mean_signed_distance_error: candidate.cadence.cadence_mean_signed_distance_error,
        cadence_max_abs_distance_error: candidate.cadence.cadence_max_abs_distance_error,
        cadence_block_hit_rate: candidate.cadence.cadence_block_hit_rate,
        cadence_within_tolerance_rate: candidate.cadence.cadence_within_tolerance_rate,
        cadence_pass: candidate.cadence.cadence_pass,
        cadence_samples: candidate.cadence.cadence_samples.clone(),
        full_cadence_start_tick: None,
        full_cadence_pairs: None,
        full_cadence_mean_abs_distance_error: None,
        full_cadence_mean_signed_distance_error: None,
        full_cadence_max_abs_distance_error: None,
        full_cadence_block_hit_rate: None,
        full_cadence_within_tolerance_rate: None,
        full_cadence_longest_hit_run: None,
        full_cadence_average_speed: None,
        full_cadence_distance: None,
        full_cadence_first_miss: None,
        full_cadence_min_hit_margin: None,
        full_cadence_mean_hit_margin: None,
        full_cadence_min_endpoint_boundary_margin: None,
        full_cadence_mean_endpoint_boundary_margin: None,
        full_cadence_samples: None,
        long_window_start_tick: None,
        long_average_vx: None,
        long_mean_vx_error: None,
        long_std_vx: None,
        long_average_distance_vx: None,
        suffix_average_vx: None,
        suffix_mean_vx_error: None,
        suffix_std_vx: None,
        suffix_average_distance_vx: None,
        first_ticks: None,
        prefix_cells: layout_cells_description(&prefix.cells),
        cycle_cells: layout_cells_description(&cycle.cells),
        note: cycle.note.clone(),
    }
}

fn full_result_row(
    candidate: &EarlyCandidate,
    prefix: &PrefixSpec,
    cycle: &CycleSpec,
    pass: &str,
    score: f64,
    full: &FullCadence,
    long: &WindowMetrics,
    suffix: Option<&WindowMetrics>,
    sim: &Simulation,
) -> ResultRow {
    ResultRow {
        id: candidate.id.clone(),
        pass: pass.to_string(),
        score,
        early_score: candidate.early_score,
        prefix_label: prefix.label.clone(),
        prefix_length: prefix.cells.len(),
        backbone: cycle.name.clone(),
        proven: cycle.proven,
        start_offset: candidate.start_offset,
        entity_id_mod4: candidate.entity_id_mod4,
        initial_tick_count: candidate.initial_tick_count,
        period: cycle.cells.len(),
        cadence_start_tick: candidate.cadence.cadence_start_tick,
        cadence_pairs: candidate.cadence.cadence_pairs,
        cadence_mean_abs_distance_error: candidate.cadence.cadence_mean_abs_distance_error,
        cadence_mean_signed_distance_error: candidate.cadence.cadence_mean_signed_distance_error,
        cadence_max_abs_distance_error: candidate.cadence.cadence_max_abs_distance_error,
        cadence_block_hit_rate: candidate.cadence.cadence_block_hit_rate,
        cadence_within_tolerance_rate: candidate.cadence.cadence_within_tolerance_rate,
        cadence_pass: candidate.cadence.cadence_pass,
        cadence_samples: candidate.cadence.cadence_samples.clone(),
        full_cadence_start_tick: Some(full.full_cadence_start_tick),
        full_cadence_pairs: Some(full.full_cadence_pairs),
        full_cadence_mean_abs_distance_error: Some(full.full_cadence_mean_abs_distance_error),
        full_cadence_mean_signed_distance_error: Some(full.full_cadence_mean_signed_distance_error),
        full_cadence_max_abs_distance_error: Some(full.full_cadence_max_abs_distance_error),
        full_cadence_block_hit_rate: Some(full.full_cadence_block_hit_rate),
        full_cadence_within_tolerance_rate: Some(full.full_cadence_within_tolerance_rate),
        full_cadence_longest_hit_run: Some(full.full_cadence_longest_hit_run),
        full_cadence_average_speed: Some(full.full_cadence_average_speed),
        full_cadence_distance: Some(full.full_cadence_distance),
        full_cadence_first_miss: full.full_cadence_first_miss.clone(),
        full_cadence_min_hit_margin: Some(full.full_cadence_min_hit_margin),
        full_cadence_mean_hit_margin: Some(full.full_cadence_mean_hit_margin),
        full_cadence_min_endpoint_boundary_margin: Some(
            full.full_cadence_min_endpoint_boundary_margin,
        ),
        full_cadence_mean_endpoint_boundary_margin: Some(
            full.full_cadence_mean_endpoint_boundary_margin,
        ),
        full_cadence_samples: Some(full.full_cadence_samples.clone()),
        long_window_start_tick: long.long_window_start_tick,
        long_average_vx: Some(long.average_vx),
        long_mean_vx_error: Some(long.mean_vx_error),
        long_std_vx: Some(long.std_vx),
        long_average_distance_vx: Some(long.average_distance_vx),
        suffix_average_vx: suffix.map(|value| value.average_vx),
        suffix_mean_vx_error: suffix.map(|value| value.mean_vx_error),
        suffix_std_vx: suffix.map(|value| value.std_vx),
        suffix_average_distance_vx: suffix.map(|value| value.average_distance_vx),
        first_ticks: Some(first_ticks(sim)),
        prefix_cells: layout_cells_description(&prefix.cells),
        cycle_cells: layout_cells_description(&cycle.cells),
        note: cycle.note.clone(),
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

fn write_csv(path: &Path, rows: &[ResultRow]) -> Result<(), String> {
    let columns = [
        "rank",
        "id",
        "pass",
        "score",
        "backbone",
        "proven",
        "prefixLabel",
        "prefixLength",
        "period",
        "startOffset",
        "entityIdMod4",
        "cadenceStartTick",
        "cadencePairs",
        "cadenceMeanAbsDistanceError",
        "cadenceMaxAbsDistanceError",
        "cadenceBlockHitRate",
        "cadenceWithinToleranceRate",
        "fullCadenceStartTick",
        "fullCadencePairs",
        "fullCadenceMeanAbsDistanceError",
        "fullCadenceMeanSignedDistanceError",
        "fullCadenceMaxAbsDistanceError",
        "fullCadenceBlockHitRate",
        "fullCadenceWithinToleranceRate",
        "fullCadenceLongestHitRun",
        "fullCadenceAverageSpeed",
        "fullCadenceDistance",
        "fullCadenceMinHitMargin",
        "fullCadenceMeanHitMargin",
        "fullCadenceMinEndpointBoundaryMargin",
        "fullCadenceMeanEndpointBoundaryMargin",
        "longWindowStartTick",
        "longAverageVX",
        "longMeanVXError",
        "longStdVX",
        "longAverageDistanceVX",
        "suffixAverageVX",
        "suffixMeanVXError",
        "suffixStdVX",
        "suffixAverageDistanceVX",
    ];
    let mut lines = Vec::with_capacity(rows.len() + 1);
    lines.push(columns.join(","));
    for (index, row) in rows.iter().enumerate() {
        let values = columns
            .iter()
            .map(|column| csv_escape(&csv_value(row, column, index + 1)))
            .collect::<Vec<_>>();
        lines.push(values.join(","));
    }
    fs::write(path, format!("{}\n", lines.join("\n")))
        .map_err(|error| format!("Failed to write CSV: {error}"))
}

fn csv_value(row: &ResultRow, column: &str, rank: usize) -> String {
    match column {
        "rank" => rank.to_string(),
        "id" => row.id.clone(),
        "pass" => row.pass.clone(),
        "score" => row.score.to_string(),
        "backbone" => row.backbone.clone(),
        "proven" => row.proven.to_string(),
        "prefixLabel" => row.prefix_label.clone(),
        "prefixLength" => row.prefix_length.to_string(),
        "period" => row.period.to_string(),
        "startOffset" => row.start_offset.to_string(),
        "entityIdMod4" => row.entity_id_mod4.to_string(),
        "cadenceStartTick" => row.cadence_start_tick.to_string(),
        "cadencePairs" => row.cadence_pairs.to_string(),
        "cadenceMeanAbsDistanceError" => row.cadence_mean_abs_distance_error.to_string(),
        "cadenceMaxAbsDistanceError" => row.cadence_max_abs_distance_error.to_string(),
        "cadenceBlockHitRate" => row.cadence_block_hit_rate.to_string(),
        "cadenceWithinToleranceRate" => row.cadence_within_tolerance_rate.to_string(),
        "fullCadenceStartTick" => option_usize(row.full_cadence_start_tick),
        "fullCadencePairs" => option_usize(row.full_cadence_pairs),
        "fullCadenceMeanAbsDistanceError" => option_f64(row.full_cadence_mean_abs_distance_error),
        "fullCadenceMeanSignedDistanceError" => {
            option_f64(row.full_cadence_mean_signed_distance_error)
        }
        "fullCadenceMaxAbsDistanceError" => option_f64(row.full_cadence_max_abs_distance_error),
        "fullCadenceBlockHitRate" => option_f64(row.full_cadence_block_hit_rate),
        "fullCadenceWithinToleranceRate" => option_f64(row.full_cadence_within_tolerance_rate),
        "fullCadenceLongestHitRun" => option_usize(row.full_cadence_longest_hit_run),
        "fullCadenceAverageSpeed" => option_f64(row.full_cadence_average_speed),
        "fullCadenceDistance" => option_f64(row.full_cadence_distance),
        "fullCadenceMinHitMargin" => option_f64(row.full_cadence_min_hit_margin),
        "fullCadenceMeanHitMargin" => option_f64(row.full_cadence_mean_hit_margin),
        "fullCadenceMinEndpointBoundaryMargin" => {
            option_f64(row.full_cadence_min_endpoint_boundary_margin)
        }
        "fullCadenceMeanEndpointBoundaryMargin" => {
            option_f64(row.full_cadence_mean_endpoint_boundary_margin)
        }
        "longWindowStartTick" => option_usize(row.long_window_start_tick),
        "longAverageVX" => option_f64(row.long_average_vx),
        "longMeanVXError" => option_f64(row.long_mean_vx_error),
        "longStdVX" => option_f64(row.long_std_vx),
        "longAverageDistanceVX" => option_f64(row.long_average_distance_vx),
        "suffixAverageVX" => option_f64(row.suffix_average_vx),
        "suffixMeanVXError" => option_f64(row.suffix_mean_vx_error),
        "suffixStdVX" => option_f64(row.suffix_std_vx),
        "suffixAverageDistanceVX" => option_f64(row.suffix_average_distance_vx),
        _ => String::new(),
    }
}

fn option_usize(value: Option<usize>) -> String {
    value.map(|value| value.to_string()).unwrap_or_default()
}

fn option_f64(value: Option<f64>) -> String {
    value.map(|value| value.to_string()).unwrap_or_default()
}

fn format_number(value: Option<f64>, digits: usize) -> String {
    value
        .map(|number| format!("{number:.digits$}"))
        .unwrap_or_default()
}

fn markdown_table(rows: &[ResultRow]) -> String {
    let header = "| Rank | Pass | Backbone | Prefix | StartX | CadenceStart | FullHit | FullAvgSpeed | HitMargin | BoundaryMargin | FullMeanErr2gt | FullMaxErr2gt | Score |";
    let sep = "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|";
    let body = rows
        .iter()
        .enumerate()
        .map(|(index, row)| {
            format!(
                "| {} | {} | `{}` | `{}` | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
                index + 1,
                row.pass,
                row.backbone,
                row.prefix_label,
                format_number(Some(row.start_offset), 5),
                row.cadence_start_tick,
                format_number(row.full_cadence_block_hit_rate, 6),
                format_number(row.full_cadence_average_speed, 9),
                format_number(row.full_cadence_min_hit_margin, 6),
                format_number(row.full_cadence_min_endpoint_boundary_margin, 6),
                format_number(row.full_cadence_mean_abs_distance_error, 6),
                format_number(row.full_cadence_max_abs_distance_error, 6),
                format_number(Some(row.score), 3),
            )
        })
        .collect::<Vec<_>>();
    [vec![header.to_string(), sep.to_string()], body]
        .concat()
        .join("\n")
}

fn markdown_early_table(rows: &[ResultRow]) -> String {
    let header = "| Rank | Pass | Backbone | Prefix | StartX | CadenceStart | EarlyHit | EarlyWithin | EarlyMeanErr2gt | EarlyMaxErr2gt | Score |";
    let sep = "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|";
    let body = rows
        .iter()
        .enumerate()
        .map(|(index, row)| {
            format!(
                "| {} | {} | `{}` | `{}` | {} | {} | {} | {} | {} | {} | {} |",
                index + 1,
                row.pass,
                row.backbone,
                row.prefix_label,
                format_number(Some(row.start_offset), 5),
                row.cadence_start_tick,
                format_number(Some(row.cadence_block_hit_rate), 6),
                format_number(Some(row.cadence_within_tolerance_rate), 6),
                format_number(Some(row.cadence_mean_abs_distance_error), 6),
                format_number(Some(row.cadence_max_abs_distance_error), 6),
                format_number(Some(row.score), 3),
            )
        })
        .collect::<Vec<_>>();
    [vec![header.to_string(), sep.to_string()], body]
        .concat()
        .join("\n")
}

fn write_summary(path: &Path, payload: &SearchPayload, args: &Args) -> Result<(), String> {
    let top_rows = payload
        .results
        .iter()
        .take(args.top)
        .cloned()
        .collect::<Vec<_>>();
    let strong_rows = payload
        .results
        .iter()
        .filter(|row| row.pass == "strong")
        .take(args.top)
        .cloned()
        .collect::<Vec<_>>();
    let proven_rows = payload
        .results
        .iter()
        .filter(|row| row.proven)
        .take(args.top)
        .cloned()
        .collect::<Vec<_>>();
    let is_early_mode = matches!(args.mode, Mode::Early);
    let top_table = if is_early_mode {
        markdown_early_table(&top_rows)
    } else {
        markdown_table(&top_rows)
    };
    let strong_table = if is_early_mode {
        "Early-only mode does not classify full-run strong candidates.".to_string()
    } else if strong_rows.is_empty() {
        "No strong candidates found.".to_string()
    } else {
        markdown_table(&strong_rows)
    };
    let proven_table = if is_early_mode {
        markdown_early_table(&proven_rows)
    } else if proven_rows.is_empty() {
        "No candidates on the proven `W3-I_D3-B` backbone passed the early cadence filter."
            .to_string()
    } else {
        markdown_table(&proven_rows)
    };

    let markdown = vec![
        "# Launch-Aware Item Waterway Search 1.17.1".to_string(),
        String::new(),
        format!(
            "Generated by `cargo run --release --` in `{:?}` mode. Evaluated `{}` launch states, kept `{}` early candidates, deduped to `{}`, and long-verified `{}` candidates.",
            args.mode, payload.evaluated, payload.early_kept, payload.early_deduped, payload.long_verified
        ),
        String::new(),
        "Launch assumption: a moving slime block from a piston has already collided with the item, so the modeled initial horizontal velocity is `vx=+1.0`. This matches the modern Minecraft source path where `PistonMovingBlockEntity.moveCollidedEntities()` overwrites the moved-axis velocity for non-player entities on slime collision, and `Entity.updateFluidInteraction()` applies water current with `0.014`.".to_string(),
        String::new(),
        format!(
            "Early hard target: some 2gt cadence phase must start at or before tick 5, with `{}` consecutive two-tick samples, each within `{}` block of 1.0 and each crossing exactly one block.",
            args.cadence_pairs, args.cadence_tolerance
        ),
        String::new(),
        format!(
            "Full model target: `{}` non-overlapping 2gt samples after the chosen cadence start, normally `{}`gt / about `{}` blocks. Primary metric is `floor(x[t+2])-floor(x[t]) == 1` hit rate.",
            args.full_cadence_pairs,
            args.full_cadence_pairs * 2,
            args.full_cadence_pairs
        ),
        String::new(),
        "## Top Overall".to_string(),
        String::new(),
        if top_rows.is_empty() {
            "No candidates passed the early cadence filter.".to_string()
        } else {
            top_table
        },
        String::new(),
        "## Strong".to_string(),
        String::new(),
        strong_table,
        String::new(),
        "## Proven Backbone Only".to_string(),
        String::new(),
        proven_table,
        String::new(),
        "## Build Notes".to_string(),
        String::new(),
        "- `D*` prefix cells are dry cells over the named floor (`N` normal, `I` packed ice, `B` blue ice, `S` slime).".to_string(),
        "- `R2*` / `R3*` prefix cells are real reverse-water gradients with the source at the right end; single-cell reverse water is intentionally not modeled because it has no source-derived flow direction.".to_string(),
        "- Dry glow lichen (`waterlogged=false`) may be used as the lane-internal non-colliding water blocker for dry gap cells. Waterlogged glow lichen should only be used where the model intentionally calls for source/still water.".to_string(),
        String::new(),
    ]
    .join("\n");

    fs::write(path, markdown).map_err(|error| format!("Failed to write markdown summary: {error}"))
}

fn constants_output() -> ConstantsOutput {
    ConstantsOutput {
        width: WIDTH,
        height: HEIGHT,
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
    }
}

fn write_json(path: &Path, payload: &SearchPayload, args: &Args) -> Result<(), String> {
    let generated_at = OffsetDateTime::now_utc()
        .format(&Rfc3339)
        .map_err(|error| format!("Failed to format timestamp: {error}"))?;
    let data = JsonOutput {
        generated_at,
        args: args.clone(),
        constants: constants_output(),
        evaluated: payload.evaluated,
        early_kept: payload.early_kept,
        early_deduped: payload.early_deduped,
        long_verified: payload.long_verified,
        top: payload.results.iter().take(args.top).cloned().collect(),
    };
    let json = serde_json::to_string_pretty(&data)
        .map_err(|error| format!("Failed to serialize JSON: {error}"))?;
    fs::write(path, format!("{}\n", json)).map_err(|error| format!("Failed to write JSON: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_args_bumps_full_ticks_for_full_cadence() {
        let argv = vec![
            "--ticks".to_string(),
            "20".to_string(),
            "--cadence-pairs".to_string(),
            "4".to_string(),
            "--long-window".to_string(),
            "5".to_string(),
            "--full-cadence-pairs".to_string(),
            "12".to_string(),
        ];
        let ParsedArgs::Run(args) = parse_args(&argv).expect("args should parse") else {
            panic!("expected runnable args");
        };
        assert_eq!(args.ticks, 33);
    }

    #[test]
    fn fluid_tracker_uses_item_height_threshold() {
        let layout = Layout::new(&[cell(Some(0.1), 0, Floor::Normal, Some(1))], &[]);
        let tracker = update_fluid_tracker(&layout, 0.0, 0.0, true);
        assert!((tracker.height - 0.1).abs() < 1.0e-12);
        assert!(tracker.is_in_fluid());
        assert!(!tracker.applies_underwater_movement());
    }

    #[test]
    fn simulate_applies_current_once_after_movement() {
        let layout = Layout::new(
            &one_way_water(2, 1, &[Floor::Normal], false),
            &dry_gap(1, &[Floor::Normal]),
        );
        let sim = simulate(
            &layout,
            &SimConfig {
                ticks: 1,
                start_x: 0.1,
                start_y: 0.0,
                start_vx: 0.0,
                start_vy: 0.0,
                entity_id_mod4: 3,
                initial_tick_count: 0,
                start_on_ground: Some(true),
            },
        );
        assert!((sim.xs[1] - 0.1).abs() < 1.0e-12);
        assert!((sim.ys[1] - BUOYANCY).abs() < 1.0e-9);
        assert!((sim.vxs[1] - WATER_PUSH).abs() < 1.0e-12);
        assert!((sim.vys[1] - BUOYANCY * VERTICAL_MOVEMENT_DAMPING).abs() < 1.0e-9);
    }

    #[test]
    fn search_small_case_stays_empty_with_stricter_fluid_ordering() {
        let args = Args {
            out: PathBuf::from("artifacts/test"),
            mode: Mode::Early,
            ticks: 17,
            top: 5,
            max_prefix: 2,
            cadence_pairs: 4,
            cadence_tolerance: 0.075,
            long_window: 10,
            start_samples: 3,
            keep_weak: false,
            min_early_block_hit_rate: 0.8,
            early_limit: 5,
            long_limit: 0,
            dedupe_long: true,
            full_cadence_pairs: 4,
            full_cadence_tolerance: 0.05,
            fixed_start_offsets: None,
            start_y: 0.0,
            start_vx: 1.0,
            start_vy: 0.0,
            start_on_ground: true,
        };
        let payload = search(&args);
        assert_eq!(payload.evaluated, 27972);
        assert!(payload.results.is_empty());
    }
}
