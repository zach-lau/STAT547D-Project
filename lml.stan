functions {
    matrix rbf(matrix x, vector l, real sf2, real sn2){
        // Compute rbf kernel
        int n = rows(x);
        int d = cols(x);
        // print("x ", x);
        matrix[n, d] x_scaled = diag_post_multiply(x, 1 ./ l);
        // Use dot product identity
        // print("x scaled", x_scaled);
        vector[n] sq_norms = rows_dot_self(x_scaled);
        // print("norms ", sq_norms);
        matrix[n, n] squared_distance = rep_matrix(sq_norms, n) + 
            rep_matrix(sq_norms', n) - 2*(x_scaled * x_scaled');
        // print("distance ", squared_distance);
        matrix[n, n] K = sf2*exp(-0.5*squared_distance) + 
            sn2*diag_matrix(rep_vector(1,n));
        // print("K ", K);
        // Force symmetry
        for (i in 1:n){
        for (j in i:n){
            real val = 0.5 * (K[i,j] + K[j,i]);
            K[i,j] = val;
            K[j,i] = val;
        }
        }
        return K;
    }
    real lml(matrix x, vector y, vector l, real sf2, real sn2){
        int n = rows(x);
        matrix[n,n] K = rbf(x, l, sf2, sn2);
        // print("K ", K);
        matrix[n,n] L = cholesky_decompose(K);
        vector[n] a = mdivide_left_tri_low(L, y);
        a = mdivide_right_tri_low(a',L)';
        real fit = dot_self(a);
        real comp = 2*sum(log(diagonal(L)));
        return -0.5*(fit + comp);
    }
}

data {
    int<lower=1> n; // Number of observations
    int<lower=1> d; // Dimension of data
    matrix[n, d] x;
    vector[n] y;
    real sn2; // noise
    real bound;
}

parameters {
    vector<lower=-bound, upper=bound>[d+1] theta; // 
}

transformed parameters {
    vector[d] l = exp(segment(theta, 1, d));
    real sf2 = exp(theta[d+1]);
}

model {
    theta ~ normal(0, 1);
    target += lml(x, y, l, sf2, sn2);
}
