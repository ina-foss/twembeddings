use std::boxed::Box;
use std::collections::HashMap;
use std::error::Error;

use clap::Clap;
use serde::Deserialize;

use crate::cli_utils::acquire_progress_indicator;

#[derive(Debug, Deserialize)]
struct VocRecord {
    token: String,
    df: usize,
    idf: f64,
}

#[derive(Clap, Debug)]
#[clap(about = "Find tweet nearest neighbors contained within a given window.")]
pub struct Opts {
    voc_input: String,
    column: String,
    input: String,
    #[clap(long)]
    total: Option<u64>,
    #[clap(long)]
    tsv: bool,
}

pub fn run(cli_args: &Opts) -> Result<(), Box<dyn Error>> {
    let mut rdr = csv::Reader::from_path(&cli_args.voc_input)?;

    // let mut wtr = csv::Writer::from_writer(std::io::stdout());

    let bar = acquire_progress_indicator("Compiling vocabulary", None);

    // 1. Compiling vocabulary
    let mut vocabulary: HashMap<String, f64> = HashMap::new();

    for result in rdr.deserialize() {
        bar.inc(1);
        let record: VocRecord = result?;
        vocabulary.insert(record.token, record.idf);
    }

    bar.finish_at_current_pos();

    eprintln!("Found {:?} distinct tokens.", vocabulary.len());

    Ok(())
}
