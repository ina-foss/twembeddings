use chrono::NaiveDateTime;
use clap::Clap;
use compound_duration::format_dhms;
use std::boxed::Box;
use std::cmp;
use std::collections::HashMap;
use std::error::Error;

use crate::cli_utils::{acquire_progress_indicator, get_column_index};

const REGULAR_DATE_FORMAT: &str = "%Y-%m-%d %H:%M:%S";

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

pub fn parse_to_day(date: &str) -> Result<String, String> {
    let datetime =
        NaiveDateTime::parse_from_str(date, REGULAR_DATE_FORMAT).or(Err("Unknown date format!"))?;
    let day = datetime.format("%Y-%m-%d").to_string();
    Ok(day)
}

pub fn run(cli_args: &Opts) -> Result<(), Box<dyn Error>> {
    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(if cli_args.tsv { b'\t' } else { b',' })
        .from_path(&cli_args.truth)?;

    let bar = acquire_progress_indicator("Indexing truth", None);

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
        "Indexed {:?} labeled tweets as truth, arranged in {:?} clusters.\n",
        truth.len(),
        truth_clusters.len()
    );

    let mut rdr = csv::ReaderBuilder::new().from_path(&cli_args.predicted)?;

    let headers = rdr.headers()?;

    let id_column_index = get_column_index(&headers, "id")?;
    let thread_column_index = get_column_index(&headers, "thread_id")?;
    let date_column_index = get_column_index(&headers, "created_at")?;

    let bar = acquire_progress_indicator("Processing predictions", cli_args.total);

    let mut predicted: HashMap<u64, usize> = HashMap::new();
    let mut predicted_clusters_sizes: HashMap<usize, (usize, String, String)> = HashMap::new();

    let mut start_date: String = String::new();
    let mut end_date: String = String::new();

    for result in rdr.records() {
        bar.inc(1);

        let record = result?;

        let id: u64 = record[id_column_index].parse()?;
        let thread_id: usize = record[thread_column_index].parse()?;
        let tweet_date: String = record[date_column_index].to_string();
        if bar.position() == 1 {
            start_date = tweet_date.clone();
        }
        end_date = tweet_date.clone();
        // We don't consider extraneous tweets
        if !truth.contains_key(&id) {
            continue;
        }

        predicted.insert(id, thread_id);
        predicted_clusters_sizes
            .entry(thread_id)
            .and_modify(|e| *e = (e.0 + 1, e.1.clone(), tweet_date.clone()))
            .or_insert((1, tweet_date.clone(), tweet_date.clone()));
    }

    bar.finish_at_current_pos();

    eprintln!(
        "Indexed {:?} tweets as predictions, arranged in {:?} clusters.\n",
        predicted.len(),
        predicted_clusters_sizes.len()
    );

    // Producing statistics about the length of events

    let bar = acquire_progress_indicator(
        "Running descriptive statistics",
        Some(predicted_clusters_sizes.len() as u64),
    );

    let start_day = parse_to_day(&start_date)?;
    let end_day = parse_to_day(&end_date)?;
    let mut max_duration = 0;
    let mut min_duration = 1000;
    let mut sum_duration = 0;
    let mut events_starting_on_start_date_count = 0;
    let mut events_ending_on_end_date_count = 0;
    let mut events_covering_whole_period_count = 0;
    let mut nb_clusters = 0;

    for (count, first_tweet_date, last_tweet_date) in predicted_clusters_sizes.values() {
        bar.inc(1);
        if count <= &1 {
            continue
        };
        nb_clusters += 1;
        if parse_to_day(&first_tweet_date)? == start_day {
            events_starting_on_start_date_count += 1;
            if parse_to_day(&last_tweet_date)? == end_day {
                events_covering_whole_period_count += 1;
            }
        }
        if parse_to_day(&last_tweet_date)? == end_day {
            events_ending_on_end_date_count += 1;
        }

        let first_datetime = NaiveDateTime::parse_from_str(first_tweet_date, REGULAR_DATE_FORMAT)
            .or(Err("Unknown date format!"))?;
        let last_datetime = NaiveDateTime::parse_from_str(last_tweet_date, REGULAR_DATE_FORMAT)
            .or(Err("Unknown date format!"))?;
        let duration = last_datetime
            .signed_duration_since(first_datetime)
            .num_seconds();
        max_duration = cmp::max(max_duration, duration);
        min_duration = cmp::min(min_duration, duration);
        sum_duration += duration;
    }

    bar.finish_at_current_pos();
    eprintln!("Stats:");
    eprintln!("  - Nb events bigger than 1 tweet: {}", nb_clusters);
    eprintln!("  - Min event length: {}", format_dhms(min_duration));
    eprintln!("  - Max event length: {}", format_dhms(max_duration));
    eprintln!(
        "  - Mean event length: {}",
        format_dhms((sum_duration as f64 / nb_clusters as f64) as usize)
    );
    eprintln!(
        "  - nb events starting on 1st day: {}",
        events_starting_on_start_date_count
    );
    eprintln!(
        "  - % events starting on 1st day: {:.4}",
        (events_starting_on_start_date_count as f64) / (nb_clusters as f64)
    );
    eprintln!(
        "  - nb events ending on last day: {}",
        events_ending_on_end_date_count
    );
    eprintln!(
        "  - % events ending on last day:    {:.4}",
        (events_ending_on_end_date_count as f64) / (nb_clusters as f64)
    );
    eprintln!(
        "  - nb events covering the whole period: {}",
        events_covering_whole_period_count
    );
    eprintln!(
        "  - % events covering the whole period:  {:.4} \n",
        events_covering_whole_period_count as f64 / (nb_clusters as f64)
    );

    // Running the actual evaluation using best matching scheme
    let bar = acquire_progress_indicator("Running evaluation", Some(truth_clusters.len() as u64));

    let mut precision = 0.0;
    let mut recall = 0.0;
    let mut f1 = 0.0;
    let mut n: usize = 0;

    for truth_cluster in truth_clusters.values() {
        bar.inc(1);

        let mut candidates: HashMap<usize, usize> = HashMap::new();
        let mut truth_cluster_size: usize = 0;

        for truth_id in truth_cluster {
            match predicted.get(truth_id) {
                Some(&candidate_thread_id) => {
                    candidates
                        .entry(candidate_thread_id)
                        .and_modify(|x| *x += 1)
                        .or_insert(1);

                    truth_cluster_size += 1;
                }
                None => {
                    continue;
                }
            }
        }

        if truth_cluster_size == 0 {
            continue;
        }

        // Only adding to n if cluster contains any valid tweet
        n += 1;

        let best = candidates
            .iter()
            .map(|(thread_id, true_positives)| {
                let matching_cluster_size = predicted_clusters_sizes[thread_id].0;

                let false_positives = (matching_cluster_size - true_positives) as f64;
                let false_negatives = (truth_cluster_size - true_positives) as f64;

                let true_positives = *true_positives as f64;

                let local_precision = true_positives / (true_positives + false_positives);
                let local_recall = true_positives / (true_positives + false_negatives);
                let local_f1 =
                    2.0 * local_precision * local_recall / (local_precision + local_recall);

                (local_precision, local_recall, local_f1)
            })
            .max_by(|x, y| x.2.partial_cmp(&y.2).unwrap())
            .unwrap();

        precision += best.0;
        recall += best.1;
        f1 += best.2;
    }

    let n = n as f64;

    precision /= n;
    recall /= n;
    f1 /= n;

    bar.finish_at_current_pos();

    eprintln!("Results:");
    eprintln!("  - Precision: {:?}", precision);
    eprintln!("  - Recall:    {:?}", recall);
    eprintln!("  - F1 score:  {:?}", f1);

    Ok(())
}
