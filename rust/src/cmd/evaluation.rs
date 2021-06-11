use std::boxed::Box;
use std::collections::HashMap;
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
    let mut truth_clusters: HashMap<usize, Vec<u64>> = HashMap::new();

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
            let cluster = truth_clusters.entry(label).or_default();
            cluster.push(id);
        }
    }

    bar.finish_at_current_pos();

    eprintln!(
        "Indexed {:?} labeled tweets as truth, arranged in {:?} clusters.",
        truth.len(),
        truth_clusters.len()
    );

    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(if cli_args.tsv { b'\t' } else { b',' })
        .from_path(&cli_args.predicted)?;

    let headers = rdr.headers()?;

    let id_column_index = get_column_index(&headers, "id")?;
    let thread_column_index = get_column_index(&headers, "thread_id")?;

    let bar = acquire_progress_indicator("Processing predictions", cli_args.total);

    let mut predicted: HashMap<u64, usize> = HashMap::new();
    let mut predicted_clusters: HashMap<usize, Vec<u64>> = HashMap::new();

    for result in rdr.records() {
        bar.inc(1);

        let record = result?;

        let id: u64 = record[id_column_index].parse()?;
        let thread_id: usize = record[thread_column_index].parse()?;

        // We don't consider extraneous tweets
        if !truth.contains_key(&id) {
            continue;
        }

        predicted.insert(id, thread_id);
        let cluster = predicted_clusters.entry(thread_id).or_default();
        cluster.push(id);
    }

    bar.finish_at_current_pos();

    eprintln!(
        "Indexed {:?} tweets as predictions, arranged in {:?} clusters.",
        predicted.len(),
        predicted_clusters.len()
    );

    Ok(())
}
