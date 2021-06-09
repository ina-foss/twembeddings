use std::boxed::Box;
use std::collections::HashMap;
use std::collections::HashSet;
use std::error::Error;

use clap::Clap;

use crate::cli_utils::{acquire_progress_indicator, get_column_index};

#[derive(Clap, Debug)]
#[clap(about = "Evaluate a clustering result.")]
pub struct Opts {
    truth: String,
    predicted: String,
    #[clap(long)]
    total: Option<u64>,
    #[clap(long)]
    tsv: bool,
}

pub fn run(cli_args: &Opts) -> Result<(), Box<dyn Error>> {
    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(if cli_args.tsv { b'\t' } else { b',' })
        .from_path(&cli_args.truth)?;

    let bar = acquire_progress_indicator("Indexing truth", cli_args.total);

    let headers = rdr.headers()?;

    let id_column_index = get_column_index(&headers, "id")?;
    let label_column_index = get_column_index(&headers, "label")?;

    let mut truth: HashMap<u64, usize> = HashMap::new();
    let mut distinct_truth_labels: HashSet<usize> = HashSet::new();

    for result in rdr.records() {
        bar.inc(1);

        let record = result?;
        let id: u64 = record[id_column_index].parse()?;
        let label_option: Option<usize> = record[label_column_index]
            .split('.')
            .next()
            .unwrap()
            .parse()
            .ok();

        if let Some(label) = label_option {
            truth.insert(id, label);
            distinct_truth_labels.insert(label);
        }
    }

    bar.finish_at_current_pos();

    eprintln!(
        "Indexed {:?} labeled tweets as truth, arranged in {:?} clusters.",
        truth.len(),
        distinct_truth_labels.len()
    );

    Ok(())
}
