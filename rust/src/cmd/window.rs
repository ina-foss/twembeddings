use std::boxed::Box;
use std::collections::HashMap;
use std::error::Error;

use chrono::NaiveDateTime;
use clap::Clap;

use crate::cli_utils::{acquire_progress_indicator, get_column_index};

const LONG_DATE_FORMAT: &str = "%a %b %d %H:%M:%S +0000 %Y";
const SHORT_DATE_FORMAT: &str = "%a %b %d %H:%M:%S";
const REGULAR_DATE_FORMAT: &str = "%Y-%m-%d %H:%M:%S";

#[derive(Clap, Debug)]
#[clap(about = "Infer the size of the window for the clustering algorithm.")]
pub struct Opts {
    input: String,
    #[clap(long)]
    raw: bool,
    #[clap(long)]
    total: Option<u64>,
    #[clap(long)]
    tsv: bool,
}

// TODO: could infer window from the vocab command in fact
pub fn run(cli_args: &Opts) -> Result<(), Box<dyn Error>> {
    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(if cli_args.tsv { b'\t' } else { b',' })
        .from_path(&cli_args.input)?;

    let bar = acquire_progress_indicator("Processing tweets", cli_args.total);

    let headers = rdr.headers()?;

    let date_column_index = get_column_index(&headers, "created_at")?;

    let mut days: HashMap<String, usize> = HashMap::new();

    for result in rdr.records() {
        bar.inc(1);

        let record = result?;

        let date_cell = &record[date_column_index];

        // Inferring date format from the string...
        let date_format = if date_cell.contains('+') {
            LONG_DATE_FORMAT
        } else if date_cell.chars().any(|c| c.is_ascii_alphabetic()) {
            SHORT_DATE_FORMAT
        } else {
            REGULAR_DATE_FORMAT
        };

        let datetime = NaiveDateTime::parse_from_str(date_cell, date_format)
            .or(Err("Unknown date format!"))?;

        let day = datetime.format("%Y-%m-%d").to_string();

        days.entry(day).and_modify(|x| *x += 1).or_insert(1);
    }

    let mut window = 0;

    for day_count in days.values() {
        window += day_count;
    }

    let window = ((window as f64) / (days.len() as f64)).floor() as usize;

    bar.finish_at_current_pos();

    if cli_args.raw {
        println!("{:?}", window);
    } else {
        eprintln!("Optimal window size is: {:?}", window);
    }
    Ok(())
}
