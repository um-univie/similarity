/// A trait for calculating similarity or distance between two entities.
/// 
/// This trait is for functions that compare two inputs and return a similarity 
/// or distance measure. Examples include cosine similarity, Euclidean distance,
/// correlation measures, etc.
/// 
/// # Examples
/// 
/// ## Pairwise similarity between vectors
/// 
/// ```
/// use similarity::{Similarity, similarity_traits::CosineSimilarity};
/// 
/// let a = [1.0, 2.0, 3.0];
/// let b = [1.0, 2.0, 3.0];
/// let result = CosineSimilarity::similarity((&a, &b));
/// assert_eq!(result, Some(1.0));
/// ```
/// 
/// ## Set similarity
/// 
/// ```
/// use similarity::{Similarity, similarity_traits::JaccardIndex};
/// use std::collections::HashSet;
/// 
/// let mut set1 = HashSet::new();
/// set1.insert(1);
/// set1.insert(2);
/// 
/// let mut set2 = HashSet::new();
/// set2.insert(2);
/// set2.insert(3);
/// 
/// let result = JaccardIndex::similarity((&set1, &set2));
/// assert!((result - 1.0/3.0).abs() < 1e-10); // intersection=1, union=3, so 1/3
/// ```
pub trait Similarity<InputType, OutputType> {
    /// Calculate the similarity or distance for the given input.
    /// 
    /// # Arguments
    /// 
    /// * `input` - The input value(s) to calculate similarity for
    /// 
    /// # Returns
    /// 
    /// The similarity or distance result of type `OutputType`
    fn similarity(input: InputType) -> OutputType;
}

/// A trait for calculating entropy measures of single entities.
/// 
/// This trait is for functions that calculate information-theoretic measures
/// like Shannon entropy, Tsallis entropy, etc. on single spectra or distributions.
/// 
/// # Examples
/// 
/// ## Shannon entropy of a spectrum
/// 
/// ```
/// use similarity::{EntropyMeasure, entropy_traits::ShannonEntropy, Spectrum, Peak};
/// 
/// let peaks = vec![
///     Peak { mz: 100.0, intensity: 0.5 },
///     Peak { mz: 200.0, intensity: 0.5 },
/// ];
/// let spectrum = Spectrum::from_peaks(peaks);
/// let entropy = ShannonEntropy::entropy(&spectrum);
/// assert!(entropy > 0.0);
/// ```
/// 
/// ## Tsallis entropy with parameter
/// 
/// ```
/// use similarity::{EntropyMeasure, entropy_traits::TsallisEntropy, Spectrum, Peak};
/// 
/// let peaks = vec![Peak { mz: 100.0, intensity: 1.0 }];
/// let spectrum = Spectrum::from_peaks(peaks);
/// let q = 2.0;
/// let entropy = TsallisEntropy::entropy((&spectrum, q));
/// assert!(entropy >= 0.0);
/// ```
pub trait EntropyMeasure<InputType, OutputType> {
    /// Calculate the entropy measure for the given input.
    /// 
    /// # Arguments
    /// 
    /// * `input` - The input value to calculate entropy for
    /// 
    /// # Returns
    /// 
    /// The entropy result of type `OutputType`
    fn entropy(input: InputType) -> OutputType;
}

/// A trait for data transformation operations.
/// 
/// This trait is for functions that transform data from one form to another,
/// such as weight factor transformations for spectral data preprocessing.
/// 
/// # Examples
/// 
/// ## Weight factor transformation
/// 
/// ```
/// use similarity::{DataTransform, transform_traits::WeightFactorTransformation};
/// 
/// let mzs = [100.0, 200.0, 300.0];
/// let intensities = [0.5, 0.3, 0.2];
/// let wf_mz = 0.5;
/// let wf_int = 0.5;
/// let result = WeightFactorTransformation::transform((&mzs, &intensities, wf_mz, wf_int));
/// assert_eq!(result.len(), 3);
/// ```
pub trait DataTransform<InputType, OutputType> {
    /// Transform the input data.
    /// 
    /// # Arguments
    /// 
    /// * `input` - The input value to transform
    /// 
    /// # Returns
    /// 
    /// The transformed result of type `OutputType`
    fn transform(input: InputType) -> OutputType;
} 