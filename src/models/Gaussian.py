import numpy as np

class Gaussian:
    def __init__(self):
        self.class_stats = {}  # Stores mu, var, and prior for each class
        self.classes_ = []

    def fit(self, X_train, y_train):
        self.classes_ = np.unique(y_train)
        for cls in self.classes_:
            X_c = X_train[y_train == cls]
            mu = np.mean(X_c, axis=0)
            var = np.var(X_c, axis=0) + 1e-6  # Avoid division by zero
            prior = len(X_c) / len(X_train)
            self.class_stats[cls] = {'mu': mu, 'var': var, 'prior': prior}

    def gaussian_pdf(self, x, mu, var):
        numerator = np.exp(-0.5 * ((x - mu) ** 2) / var)
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

    def predict_proba(self, x):
        probs = []
        for cls in self.classes_:
            stats = self.class_stats[cls]
            pdfs = self.gaussian_pdf(x, stats['mu'], stats['var'])
            log_likelihood = np.sum(np.log(pdfs + 1e-9)) + np.log(stats['prior'] + 1e-9)
            probs.append(log_likelihood)

        # Convert log-likelihoods to softmax probabilities
        probs = np.exp(probs - np.max(probs))  # For numerical stability
        probs /= np.sum(probs)
        return probs

    def predict(self, x):
        return np.argmax(self.predict_proba(x))
