use std::boxed::Box;
use std::collections::HashMap;
use std::error::Error;

use chrono::NaiveDateTime;
use clap::Clap;

use crate::cli_utils::{acquire_progress_indicator, get_column_index};

const LONG_DATE_FORMAT: &str = "%a %b %d %H:%M:%S +0000 %Y";

#[derive(Clap, Debug)]
#[clap(about = "Infer the size of the window for the clustering algorithm.")]
pub struct Opts {
    input: String,
    #[clap(long)]
    total: Option<u64>,
    #[clap(long)]
    tsv: bool,
}

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
        let datetime = NaiveDateTime::parse_from_str(date_cell, LONG_DATE_FORMAT)?;
        let day = datetime.format("%Y-%m-%d").to_string();

        days.entry(day).and_modify(|x| *x += 1).or_insert(1);
    }

    let mut window = 0;

    for day_count in days.values() {
        window += day_count;
    }

    let window = ((window as f64) / (days.len() as f64)).floor() as usize;

    bar.finish_at_current_pos();

    eprintln!("Optimal window size is: {:?}", window);

    Ok(())
}
