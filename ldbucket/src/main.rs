use anyhow::{Context, Result, bail};
use clap::{Parser, command};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use rayon::ThreadPoolBuilder;
use rayon::prelude::*;
use rust_htslib::bcf::header::HeaderRecord;
use rust_htslib::bcf::{IndexedReader, Read, Record};
use std::cmp;
use std::collections::BTreeMap;
use std::ops::Deref;
use std::sync::Arc;

// This function computes E[X_iY_iX_jY_j]
pub fn linkage_disequilibrium(genotypes1: &[f64], genotypes2: &[f64], n_samples: usize) -> f64 {
    let s = n_samples as f64;
    let (ld, ld_square) = genotypes1.iter().zip(genotypes2.iter()).fold(
        (0.0, 0.0),
        |(ld_acc, ld_square_acc), (&a, &b)| {
            let prod = a * b;
            (ld_acc + prod, ld_square_acc + prod * prod)
        },
    );
    (ld * ld - ld_square) / (s * (s - 1.0))
}

struct Bins {
    nbins: usize,
    left_edges_in_cm: Vec<f64>,
    right_edges_in_cm: Vec<f64>,
    left_edges_in_bp: Vec<f64>,
    right_edges_in_bp: Vec<f64>,
    minimum: i64,
    maximum: i64,
}

impl Bins {
    // From HapNe supplementary material
    fn hapne_default(recombination_rate: f64) -> Self {
        let nbins = 19;
        let mut left_edges_in_cm = Vec::with_capacity(nbins);
        let mut right_edges_in_cm = Vec::with_capacity(nbins);

        for i in 0..nbins {
            let i = i as f64;
            left_edges_in_cm.push(0.5 + 0.5 * i);
            right_edges_in_cm.push(1.0 + 0.5 * i);
        }
        // Transform to base pairs using x / 100 / recombination_rate
        let left_edges_in_bp = left_edges_in_cm
            .iter()
            .map(|&x| x / 100.0 / recombination_rate)
            .collect::<Vec<f64>>();
        let right_edges_in_bp = right_edges_in_cm
            .iter()
            .map(|&x| x / 100.0 / recombination_rate)
            .collect::<Vec<f64>>();
        let minimum = left_edges_in_bp[0].round() as i64;
        let maximum = right_edges_in_bp[nbins - 1].round() as i64;
        Self {
            nbins,
            left_edges_in_cm,
            right_edges_in_cm,
            left_edges_in_bp,
            right_edges_in_bp,
            minimum,
            maximum,
        }
    }
}

// The sufficient summary statistics for computing the log-likelihood later
#[derive(Debug)]
struct SufficientSummaryStats {
    pub mean: Vec<f64>,
    pub variance: Vec<f64>,
    pub n: Vec<usize>,
}

impl SufficientSummaryStats {
    pub fn iter(&self) -> impl Iterator<Item = (&f64, &f64, &usize)> {
        self.mean
            .iter()
            .zip(self.variance.iter())
            .zip(self.n.iter())
            .map(|((m, v), n)| (m, v, n))
    }
}

// We use a classic streaming algorithm to do only one pass over the data
// and have a constant memory footprint.
#[derive(Debug, Clone)]
struct StreamingStats {
    pub counts: Vec<usize>,
    pub ld: Vec<f64>,
    pub ld_square: Vec<f64>,
}
impl StreamingStats {
    pub fn new(n_bins: usize) -> Self {
        Self {
            counts: vec![0; n_bins],
            ld: vec![0.0; n_bins],
            ld_square: vec![0.0; n_bins],
        }
    }
    pub fn update(
        &mut self,
        index: usize,
        genotypes1: &[f64],
        genotypes2: &[f64],
        n_samples: usize,
    ) {
        // Compute the sufficient statistics
        let new_value = linkage_disequilibrium(genotypes1, genotypes2, n_samples);
        self.counts[index] += 1;
        let delta = new_value - self.ld[index];
        self.ld[index] += delta / self.counts[index] as f64;
        let delta2 = new_value - self.ld[index];
        self.ld_square[index] += delta * delta2;
    }
    pub fn finalize(&mut self) -> SufficientSummaryStats {
        let mut mean = self.ld.clone();
        let mut var = self.ld_square.clone();
        for i in 0..self.counts.len() {
            if self.counts[i] > 1 {
                var[i] /= self.counts[i] as f64;
            } else {
                var[i] = f64::NAN;
                mean[i] = f64::NAN;
            }
        }
        SufficientSummaryStats {
            mean,
            variance: var,
            n: self.counts.clone(),
        }
    }
}

#[derive(Debug, Clone)]
struct Contig {
    rid: u32,
    length: u64,
    start: u64,
    end: u64,
}

impl Contig {
    fn new(rid: u32, length: u64, start: u64, end: u64) -> Self {
        Contig {
            rid,
            length,
            start,
            end,
        }
    }
    fn build(
        header_records: &Vec<rust_htslib::bcf::HeaderRecord>,
        contig_name: &str,
    ) -> Result<Self> {
        // Contig names might be in the format chr1:1-1000000
        let mut split = contig_name.split(':');
        let name = split.next().unwrap_or(contig_name);
        let mut remainder = split.next().unwrap_or("");
        if remainder.is_empty() {
            remainder = "1-";
        }
        let mut remainder = remainder.split('-');
        // We take start if present, otherwise default to 1 (but zero-indexed)
        let start = remainder
            .next()
            .expect("Start position not found")
            .parse::<u64>()
            .context("Invalid start position")?;
        let start = start - 1;
        let mut rid = 0;
        for record in header_records {
            if let HeaderRecord::Contig { values, .. } = record {
                if values.get("ID").unwrap_or(&"".to_string()) == name {
                    let contig_length = values.get("length").context("Contig length not found")?;
                    if let Ok(contig_length) = contig_length.parse::<u64>() {
                        let end = remainder.next();
                        let end = match end {
                            Some(e) if !e.trim().is_empty() => {
                                e.parse::<u64>().context("Invalid end position")? - 1
                            }
                            _ => contig_length,
                        };
                        return Ok(Contig::new(rid, end - start, start, end));
                    }
                }
                rid += 1;
            }
        }
        bail!("Contig not found")
    }
}

// The key part of this algorithm is how to tradeoff memory and computation time.
// Here, I propose a solution that does a one pass over the data with two pointers
// but avoids recomputing some intermediate results with a B-tree
struct RollingMap {
    // Maps a given position to a vector of standarized genotypes
    map: BTreeMap<u64, Box<[f64]>>,
    last_position: u64,
    // Map samples to their indices
    sample_indices: Vec<usize>,
    use_precomputed_maf: bool,
    contig: Contig,
    maf_threshold: f64,
    pub(crate) n_samples: usize,
}

impl RollingMap {
    pub fn build(
        header: &rust_htslib::bcf::header::HeaderView,
        contig: Contig,
        parameters: &Cli,
    ) -> Result<Self> {
        // Create a bijection for the samples (or return an informative error)
        let sample_indices = match &parameters.sample_names {
            Some(sample_names) => {
                let mut sample_indices = Vec::with_capacity(sample_names.len());
                for name in sample_names {
                    match header.sample_id(name.as_bytes()) {
                        Some(idx) => sample_indices.push(idx),
                        None => bail!("Sample name '{}' not found in VCF header", name),
                    }
                }
                sample_indices.sort();
                if sample_indices.len() < 2 {
                    bail!("No enough samples (at least 2 required)");
                }
                if sample_indices.windows(2).any(|w| w[0] == w[1]) {
                    bail!("Duplicate sample names found in VCF header");
                }
                sample_indices
            }
            None => (0..header.sample_count() as usize).collect::<Vec<_>>(),
        };
        let n_samples = sample_indices.len();
        // Check whether a MAF column is present
        let has_maf = header.name_to_id(b"MAF").is_ok();
        if has_maf && parameters.use_precomputed_maf {
            // If MAF is present and user wants to use precomputed, skip (do nothing)
        } else if has_maf && !parameters.use_precomputed_maf {
            eprintln!(
                "Warning: MAF INFO field present in VCF, but --use-precomputed-maf not set. Using on-the-fly MAF calculation."
            );
        } else if !has_maf && parameters.use_precomputed_maf {
            bail!(
                "Requested to use precomputed MAF (--use-precomputed-maf), but no MAF INFO field found in VCF header."
            );
        }

        Ok(Self {
            map: BTreeMap::new(),
            last_position: 0,
            sample_indices,
            use_precomputed_maf: parameters.use_precomputed_maf,
            n_samples,
            maf_threshold: parameters.maf_threshold,
            contig,
        })
    }
    #[allow(clippy::borrowed_box)]
    pub fn lookup(
        &mut self,
        record: &Record,
        genotypes_buffer: &mut [f64],
    ) -> Result<(u64, Option<&Box<[f64]>>)> {
        let pos = record.pos() as u64;

        if self.map.contains_key(&pos) {
            return Ok((pos, self.map.get(&pos)));
        }
        // Try to process and insert
        if self.process(record, genotypes_buffer)?.is_some() {
            let genotypes = genotypes_buffer.to_vec().into_boxed_slice();
            self.map.insert(pos, genotypes);
            Ok((pos, self.map.get(&pos)))
        } else {
            Ok((pos, None))
        }
    }

    pub fn roll_window(
        &mut self,
        reader: &mut IndexedReader,
        genotypes_buffer: &mut [f64],
        start: u64,
        end: u64,
    ) -> Result<()> {
        let start = cmp::max(start, self.contig.start);
        let end = cmp::min(end, self.contig.end);
        // This function rolls the window to the start-end region
        // First, we split off the part of the map that is before the new window start
        self.map = self.map.split_off(&start);
        // Then, we have to fetch next records
        let _ = reader.fetch(self.contig.rid, self.last_position, Some(end));
        // Now iterate over the collected records
        for record in reader.records() {
            let record = record.context("Error while reading record")?;
            // We call lookup to process and potentially insert the record.
            let _ = self.lookup(&record, genotypes_buffer)?;
        }
        self.last_position = end;
        Ok(())
    }

    fn process(&self, record: &Record, genotypes_buffer: &mut [f64]) -> Result<Option<()>> {
        // First, we handle the case where we're using precomputed MAF
        if self.use_precomputed_maf {
            let maf = record.info(b"MAF").float().context("Error getting MAF")?;
            let maf = match maf {
                Some(maf) => *maf.deref().first().unwrap_or(&0.0) as f64,
                None => return Ok(None),
            };
            if maf.is_nan() {
                eprintln!("MAF is NaN");
                return Ok(None);
            }
            if maf < self.maf_threshold {
                return Ok(None);
            }
        }
        let n_samples = self.n_samples;
        let raw_genotypes = record.genotypes().context("Error getting genotypes")?;
        let mut total = 0;
        genotypes_buffer.fill(0.0);
        for (index, val) in genotypes_buffer.iter_mut().enumerate() {
            let i = self.sample_indices[index];
            let sample = raw_genotypes.get(i);
            for j in 0..2 {
                if let Some(gt) = sample[j].index() {
                    *val += gt as f64;
                    total += gt;
                } else {
                    return Ok(None);
                }
            }
        }
        let allele_freq = total as f64 / (2 * n_samples) as f64;
        if !self.use_precomputed_maf
            && (allele_freq < self.maf_threshold || allele_freq > (1.0 - self.maf_threshold))
        {
            return Ok(None);
        }
        // Standardize the genotypes
        let denom = (2.0 * allele_freq * (1.0 - allele_freq)).sqrt();
        for val in genotypes_buffer.iter_mut() {
            *val = (*val - 2.0 * allele_freq) / denom;
        }
        Ok(Some(()))
    }
}

#[derive(Parser, Clone)]
#[command(
    version,
    about = "Computes observed linkage disequilibrium between all pairs of variants",
    after_help = "The output is a CSV file with the following columns:
  - `contig_name`: name of the contig
  - `bin_index`: index of the bin
  - `left_bin`: left bin position in Morgan
  - `right_bin`: right bin position in Morgan
  - `mean`: estimated E[X_iX_jY_iY_j]
  - `var`: estimated Var[X_iX_jY_iY_j]
  - `N`: number of unique pairs per bin"
)]
struct Cli {
    /// Compressed and indexed VCF or BCF (better!) file
    infile: String,

    /// Number of cores (only used if more than one contig is processed)
    #[arg(long, default_value_t = 1)]
    cores: usize,

    /// Recombination rate per base pair
    #[arg(long, default_value_t = 1.0e-8)]
    recombination_rate: f64,

    /// Minor allele frequency threshold
    #[arg(long, default_value_t = 0.25)]
    maf_threshold: f64,

    /// List of contig names (optional). If not provided, all contigs will be processed.
    #[arg(long, value_delimiter = ',')]
    contig_names: Option<Vec<String>>,

    /// List of sample names (optional). If not provided, all samples will be processed.
    #[arg(long, value_delimiter = ',')]
    sample_names: Option<Vec<String>>,

    /// Whether to use a precomputed MAF TAG. Be careful, this can lead to incorrect results if the MAF TAG is not accurate.
    #[arg(long, default_value_t = false)]
    use_precomputed_maf: bool,
}

fn contig_analysis(
    contig_name: &str,
    bins: &Bins,
    parameters: Cli,
    progress_bar: ProgressBar,
) -> Result<SufficientSummaryStats> {
    // Open indexed VCF or BCF (better)
    let src = &parameters.infile;
    let mut file =
        IndexedReader::from_path(src).with_context(|| format!("Error opening file {src}"))?;
    // Get contig information
    let header = file.header();
    let header_records = header.header_records();
    let contig = Contig::build(&header_records, contig_name)?;
    // Check if the length of the contig is greater than bins.minimum
    if contig.length < (bins.minimum as u64) {
        bail!("Contig length is less than minimum bin size");
    }
    // Initialize data structures
    let mut streaming = StreamingStats::new(bins.nbins);
    // We need a second pointer to iterate across all pairs
    let mut reader =
        IndexedReader::from_path(src).with_context(|| format!("Error opening file {src}"))?;
    let _ = reader.header();
    // Initialize rolling window
    let mut rolling_window = RollingMap::build(header, contig.clone(), &parameters)?;
    let n_samples = rolling_window.n_samples;
    let mut genotypes_buffer = vec![0.0; n_samples];
    // Next, we iterate over all the SNPs
    progress_bar.set_length(100);
    progress_bar
        .set_style(ProgressStyle::with_template("{prefix} {spinner} {wide_bar} {pos}%").unwrap());
    progress_bar.set_prefix(contig_name.to_string());
    // First, we fetch the entire chromosome of interest
    let _ = file.fetch(contig.rid, contig.start, Some(contig.end));
    // and iterate across the records
    for record1_result in file.records() {
        let record1 = record1_result.context("Error while reading record")?;
        // Keep processing records in ascending order
        let (pos1, genotypes1) = {
            let (p, maybe_ref) = rolling_window.lookup(&record1, &mut genotypes_buffer)?;
            // Set progress bar as percentage of contig length
            let percent = ((p as f64 / contig.length as f64) * 100.0).clamp(0.0, 100.0) as u64;
            progress_bar.set_position(percent);
            match maybe_ref {
                None => continue,
                Some(g_ref) => (p, g_ref.clone()),
            }
        };
        let (start, end) = (pos1 + bins.minimum as u64, pos1 + bins.maximum as u64);
        // Next, we roll the window to the next position.
        // NOTE: The start of the new window should be pos1. This way, next iteration will take advantage
        // of the recorded genotypes.
        rolling_window.roll_window(&mut reader, &mut genotypes_buffer, pos1, end)?;
        // Most of the time, the second record will be in the first bin
        // and we know the bin index is monotonic increasing.
        let mut index = 0;
        for (pos2, genotypes2) in rolling_window.map.range(start..end) {
            let distance = (pos2 - pos1) as f64;
            assert!(distance >= bins.minimum as f64 && distance <= bins.maximum as f64);
            // Find current bin index
            while distance > bins.right_edges_in_bp[index] {
                index += 1;
            }
            assert!(index < bins.nbins);
            if bins.left_edges_in_bp[index] <= distance && distance <= bins.right_edges_in_bp[index]
            {
                // Update the sufficient statistics
                streaming.update(index, &genotypes1, genotypes2, n_samples);
            }
        }
    }
    progress_bar.finish_and_clear();
    // We have finished!
    Ok(streaming.finalize())
}

fn display(bins: Bins, stats: Vec<SufficientSummaryStats>, contigs: Vec<String>) {
    println!("contig_name,bin_index,left_bin,right_bin,mean,var,N");
    for (summary_stats, contig_name) in stats.into_iter().zip(contigs) {
        for (i, (mean, var, n)) in summary_stats.iter().enumerate() {
            println!(
                "{},{},{},{},{},{},{}",
                contig_name,
                i,
                bins.left_edges_in_cm[i] / 100.0,
                bins.right_edges_in_cm[i] / 100.0,
                mean,
                var,
                n
            );
        }
    }
}

fn main() -> Result<()> {
    // Read parameters from command line
    let args = Cli::parse();
    let num_threads = args.cores;
    ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global()
        .expect("Failed to set number of threads");
    // If contig_names is not specified, use all contigs:
    let contigs = match args.contig_names.clone() {
        Some(contigs) => contigs,
        None => {
            let src = &args.infile;
            let file = IndexedReader::from_path(src)
                .with_context(|| format!("Error opening file {src}"))?;
            // Get contig information
            let header = file.header();
            let n_contigs = header.contig_count();
            // Discard errors
            (0..n_contigs)
                .filter_map(|i| header.rid2name(i).ok())
                .map(|name_bytes| String::from_utf8_lossy(name_bytes).to_string())
                .collect::<Vec<String>>()
        }
    };
    let bins = Bins::hapne_default(args.recombination_rate);
    let master = Arc::new(MultiProgress::new());
    let overview = master.add(ProgressBar::new(contigs.len() as u64));
    overview.set_style(
        ProgressStyle::with_template("{bar} {pos}/{len} contigs")
            .expect("Failed to set progress bar style"),
    );
    let results: Vec<SufficientSummaryStats> = contigs
        .par_iter()
        .map(|contig| {
            let child_pb = master.add(ProgressBar::no_length());
            let stats = contig_analysis(contig, &bins, args.clone(), child_pb);
            overview.inc(1);
            match stats {
                Ok(res) => Some(res),
                Err(e) => {
                    eprintln!("Contig '{contig}': {e}");
                    None
                }
            }
        })
        .filter_map(|x| x)
        .collect();
    overview.finish_with_message("All contigs processed");
    display(bins, results, contigs);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn simulate_data(n_samples: usize) -> (Vec<f64>, Vec<f64>) {
        use rand_distr::{Distribution, Normal};
        let mut rng = rand::rng();
        let normal = Normal::new(0.0, 1.0).unwrap();
        let genotypes1: Vec<f64> = (0..n_samples).map(|_| normal.sample(&mut rng)).collect();
        let genotypes2: Vec<f64> = (0..n_samples).map(|_| normal.sample(&mut rng)).collect();
        (genotypes1, genotypes2)
    }

    #[test]
    fn test_linkage_disequilibrium() {
        use rand::Rng;
        let mut rng = rand::rng();
        let n_samples = rng.random_range(1..=100);
        let (genotypes1, genotypes2) = simulate_data(n_samples);
        let result = linkage_disequilibrium(&genotypes1, &genotypes2, n_samples);
        // Compare with the naive implementation
        let mut expected = 0.0;
        let mut acc = 0;
        for i in 0..n_samples {
            for j in (i + 1)..n_samples {
                acc += 1;
                expected += genotypes1[i] * genotypes2[i] * genotypes1[j] * genotypes2[j];
            }
        }
        expected /= acc as f64;
        let diff = expected - result;
        assert!(diff.abs() < 1e-10);
    }
    #[test]
    fn test_streaming_statistics() {
        use rand::Rng;
        let mut rng = rand::rng();
        let n_samples = rng.random_range(1..=100);
        let n_genotypes = rng.random_range(1..=10000);
        let mut ld = Vec::with_capacity(n_genotypes);
        let mut streaming = StreamingStats::new(1);
        for _ in 0..n_genotypes {
            let (genotypes1, genotypes2) = simulate_data(n_samples);
            ld.push(linkage_disequilibrium(&genotypes1, &genotypes2, n_samples));
            streaming.update(0, &genotypes1, &genotypes2, n_samples);
        }
        let sufficient_stats = streaming.finalize();
        let (mean, var, n) = sufficient_stats.iter().next().unwrap();
        assert_eq!(*n, n_genotypes);
        // Calculate mean
        let expected_mean = ld.iter().sum::<f64>() / ld.len() as f64;
        let expected_var =
            ld.iter().map(|x| (x - expected_mean).powi(2)).sum::<f64>() / ld.len() as f64;
        let diff_mean = expected_mean - mean;
        let diff_var = expected_var - var;
        assert!(diff_mean.abs() < 1e-10);
        assert!(diff_var.abs() < 1e-10);
    }
}
