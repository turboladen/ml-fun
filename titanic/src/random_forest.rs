use anyhow::Result;
use linfa::{Label, prelude::*};
use linfa_ensemble::EnsembleLearnerValidParams;
use linfa_trees::{DecisionTree, SplitQuality};
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;

/// A Random Forest Classifier that mimics sklearn's RandomForestClassifier
///
/// This uses linfa's EnsembleLearner with DecisionTree as the base estimator.
pub struct RandomForestClassifier {
    n_estimators: usize,
    max_depth: usize,
    min_samples_split: usize,
    bootstrap_proportion: f64,
    random_state: Option<u64>,
}

impl RandomForestClassifier {
    /// Create a new Random Forest Classifier with default parameters
    pub fn new() -> Self {
        Self {
            n_estimators: 100,
            max_depth: 10,
            min_samples_split: 2,
            bootstrap_proportion: 1.0,
            random_state: None,
        }
    }

    /// Set the number of trees in the forest (default: 100)
    pub fn n_estimators(mut self, n: usize) -> Self {
        self.n_estimators = n;
        self
    }

    /// Set the maximum depth of each tree (default: 10)
    pub fn max_depth(mut self, depth: usize) -> Self {
        self.max_depth = depth;
        self
    }

    /// Set the minimum number of samples required to split a node (default: 2)
    pub fn min_samples_split(mut self, n: usize) -> Self {
        self.min_samples_split = n;
        self
    }

    /// Set the proportion of samples to use for each bootstrap sample (default: 1.0)
    pub fn bootstrap_proportion(mut self, proportion: f64) -> Self {
        self.bootstrap_proportion = proportion;
        self
    }

    /// Set the random seed for reproducibility
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Fit the Random Forest on training data
    ///
    /// # Arguments
    /// * `x` - Feature matrix (rows = samples, columns = features)
    /// * `y` - Target vector (class labels)
    ///
    /// # Returns
    /// A fitted Random Forest model that can be used for prediction
    pub fn fit<L: 'static + Clone + Copy + Ord + Label>(
        &self,
        x: Array2<f64>,
        y: Array1<L>,
    ) -> Result<FittedRandomForest<L>> {
        // Create the dataset
        let dataset = Dataset::new(x, y);

        // Configure the decision tree parameters
        let tree_params = DecisionTree::params()
            .max_depth(Some(self.max_depth))
            .min_weight_split(self.min_samples_split as f32)
            .split_quality(SplitQuality::Gini);

        // Create RNG
        let rng = match self.random_state {
            Some(seed) => Xoshiro256Plus::seed_from_u64(seed),
            None => Xoshiro256Plus::from_entropy(),
        };

        // Configure the ensemble
        let ensemble_params = EnsembleLearnerValidParams {
            model_params: tree_params,
            ensemble_size: self.n_estimators,
            bootstrap_proportion: self.bootstrap_proportion,
            rng,
        };

        // Fit the ensemble
        let model = ensemble_params.fit(&dataset)?;

        Ok(FittedRandomForest { model })
    }
}

impl Default for RandomForestClassifier {
    fn default() -> Self {
        Self::new()
    }
}

/// A fitted Random Forest model that can make predictions
pub struct FittedRandomForest<L: Label> {
    model: linfa_ensemble::EnsembleLearner<DecisionTree<f64, L>>,
}

impl<L: Clone + Copy + Ord + std::hash::Hash + Eq + std::fmt::Debug + Label> FittedRandomForest<L> {
    /// Predict class labels for samples in X
    ///
    /// # Arguments
    /// * `x` - Feature matrix (rows = samples, columns = features)
    ///
    /// # Returns
    /// Array of predicted class labels
    pub fn predict(&self, x: &Array2<f64>) -> Array1<L> {
        self.model.predict(x)
    }
}
