functions {
    vector cross(vector x, array[] vector alpha, vector l, real sf2){
        // Compute the cross covariance vector
        int m = size(alpha);
        vector[m] out;
        for (i in 1:m){
            out[i] = sf2*exp(-0.5*dot_self((x - alpha[i])./l));
        }
        return out;
    }
    real gp_mean(vector k, vector weights){
        return dot_product(k,  weights);
    }
    real gp_variance(vector k, matrix Kinv, real sf2){
        return sf2 - dot_product(k, Kinv*k); // TODO change to cholesky
    }
    real logf(vector x, array[] vector alpha, vector weights, vector l, real sf2, real mu, real s2){
        // Evaluate surogate likelihood using gp mean
        int m = size(alpha);
        vector[m] k = cross(x, alpha, l, sf2);
        return gp_mean(k, weights)*sqrt(s2)+mu;
    }
}

data {
    int<lower=1> m; // Number of likelihood evaluations
    int<lower=1> d; // Dimension of data
    matrix[m, m] Kinv; // inverse matrix at data points
    array[m] vector[d+1] alpha; // Observation locations
    vector[m] gamma; // Scaled and centered log-likelhood observationso
    vector[d+1] l; // Surrogate length-scale
    real sf2; // Surrogate output-Scaled
    real mu; // Centering for gamma
    real s2; // scaling for gamma
    real bound;
}

transformed data {
    vector[m] weights = Kinv * gamma;
}

parameters {
    vector<lower=-bound, upper=bound>[d+1] theta; 
}

model {
    theta ~ normal(0, 1);
    target += logf(theta, alpha, weights, l, sf2, mu, s2);
}
