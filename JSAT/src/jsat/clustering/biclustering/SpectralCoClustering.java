/*
 * This code was contributed under the public domain. 
 */
package jsat.clustering.biclustering;

import java.util.List;
import jsat.DataSet;
import jsat.SimpleDataSet;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.DataPoint;
import jsat.clustering.Clusterer;
import jsat.clustering.KClusterer;
import jsat.clustering.kmeans.GMeans;
import jsat.clustering.kmeans.HamerlyKMeans;
import jsat.linear.DenseVector;
import jsat.linear.IndexValue;
import jsat.linear.Matrix;
import jsat.linear.SubMatrix;
import jsat.linear.TruncatedSVD;
import jsat.linear.Vec;
import jsat.utils.IntList;

/**
 *
 * @author Edward Raff
 */
public class SpectralCoClustering implements Bicluster
{
    static public enum InputNormalization 
    {
        SCALE
        {
            @Override
            protected Matrix normalize(Matrix A, DenseVector R, DenseVector C) 
            {
                return row_col_normalize(A, R, C);
            }
            
        },
        BISTOCHASTIZATION
        {
            @Override
            protected Matrix normalize(Matrix A, DenseVector R, DenseVector C) 
            {
                //Init locations to store final normalization vectors
                //make equal to no normalization at first, and accumulate after every step
                DenseVector R_tmp = R.clone();
                R_tmp.zeroOut();
                R_tmp.mutableAdd(1.0);
                DenseVector C_tmp = C.clone();
                C_tmp.zeroOut();
                C_tmp.mutableAdd(1.0);
                
                Matrix A_prev = A;
                double diff = Double.POSITIVE_INFINITY;
                int iter = 0;
                
                while(iter++ < 1000 && diff > 1e-4)
                {
                    A_prev = A;
                    A = row_col_normalize(A, R, C);
                    
                    //Compute the "norm" of the two matricies
                    //below is not quite the norm of the 2 matricies, but close 
                    //enough, we just need to know if we have converged
                    diff = 0;
                    for(int row = 0; row < A.rows(); row++)
                        diff += A.getRowView(row).pNormDist(2, A_prev.getRowView(row));
                    diff /= A.rows();
                    
                    R_tmp.mutablePairwiseMultiply(R);
                    C_tmp.mutablePairwiseMultiply(C);
                }
                
                R_tmp.copyTo(R);
                C_tmp.copyTo(C);
                
                return A;
            }
        };
        
        /**
         * Computes a normalization of the input matrix that allows a later SVD
         * step to better reveal block structure in the underlying data.
         *
         * @param A the input matrix to normalize, which will not be altered
         * @param R the location to store the row normalization matrix to apply
         * to the original matrix to get the result. This may be filled with
         * constants if no subsequent use of this should occur in the SVD
         * decomposition.
         * @param C the location to store the column normalization matrix to
         * apply to the original matrix to get the result. This may be filled
         * with constants if no subsequent use of this should occur in the SVD
         * decomposition.
         * @return A new matrix that has been normalized
         */
        abstract protected Matrix normalize(Matrix A, DenseVector R, DenseVector C);
       
    }
    
    private Clusterer baseClusterAlgo;
    
    private InputNormalization inputNormalization;

    public SpectralCoClustering() 
    {
        this(InputNormalization.SCALE);
    }

    public SpectralCoClustering(InputNormalization normalization) 
    {
        this(normalization, new GMeans(new HamerlyKMeans()));
    }
    
    public SpectralCoClustering(InputNormalization normalization, Clusterer baseCluster) 
    {
        setBaseClusterAlgo(baseCluster);
        setInputNormalization(normalization);
    }

    public void setInputNormalization(InputNormalization inputNormalization) 
    {
        this.inputNormalization = inputNormalization;
    }

    public InputNormalization getInputNormalization() 
    {
        return inputNormalization;
    }

    public void setBaseClusterAlgo(Clusterer baseClusterAlgo) 
    {
        this.baseClusterAlgo = baseClusterAlgo;
    }

    public Clusterer getBaseClusterAlgo() 
    {
        return baseClusterAlgo;
    }

    @Override
    public void bicluster(DataSet dataSet, int clusters, boolean parallel, List<List<Integer>> row_assignments, List<List<Integer>> col_assignments) 
    {
        //﻿1. Given A, form An = D_1^{−1/2} A D_2^{−1/2}
        Matrix A = dataSet.getDataMatrix();
                        
        DenseVector R = new DenseVector(A.rows());
        DenseVector C = new DenseVector(A.cols());
        
        Matrix A_n = inputNormalization.normalize(A, R, C);
        
        //﻿2. Compute l = ceil(log2 k) singular vectors of A_n, u2, . . . u_l+1 and v2, . . . v_l+1, and form the matrix Z as in (12)
        int l = (int) Math.ceil(Math.log(clusters)/Math.log(2.0));
        
        
        //A_n has r rows and c columns. We are going to make a new data matrix Z
        //Z will have (r+c) rows, and l columns. 
        SimpleDataSet Z = create_Z_dataset(A_n, l, R, C, inputNormalization);//+1 b/c we are going to skip the first SV
        
        KClusterer to_use;
        if(baseClusterAlgo instanceof KClusterer)
            to_use = (KClusterer) baseClusterAlgo;
        else
            to_use = new HamerlyKMeans();
        int[] joint_designations = to_use.cluster(Z, clusters, parallel, null);
        
        createAssignments(row_assignments, col_assignments, clusters, A, joint_designations);
        
    }

    public void bicluster(DataSet dataSet, boolean parallel, List<List<Integer>> row_assignments, List<List<Integer>> col_assignments)
    {
        //﻿1. Given A, form An = D_1^{−1/2} A D_2^{−1/2}
        Matrix A = dataSet.getDataMatrix();
                        
        DenseVector R = new DenseVector(A.rows());
        DenseVector C = new DenseVector(A.cols());
        
        Matrix A_n = inputNormalization.normalize(A, R, C);
        
        //﻿2. Compute l = ceil(log2 k) singular vectors of A_n, u2, . . . u_l+1 and v2, . . . v_l+1, and form the matrix Z as in (12)
        int k_max = Math.min(A.rows(), A.cols());
        int l = (int) Math.ceil(Math.log(k_max)/Math.log(2.0));
        
        
        SimpleDataSet Z = create_Z_dataset(A_n, l, R, C, inputNormalization);
        
        
        int[] joint_designations = baseClusterAlgo.cluster(Z, parallel, null);
        int clusters = 0;
        for(int i : joint_designations)
            clusters = Math.max(clusters, i+1);
        
        //prep label outputs
        createAssignments(row_assignments, col_assignments, clusters, A, joint_designations);
        //the bicluster labels for the rows
        //the bicluter labels for the columns
        //Now we need to prune potential false bi-clusterings that have only features or only rows
        
    }

    private SimpleDataSet create_Z_dataset(Matrix A_n, int l, DenseVector R, DenseVector C, InputNormalization inputNormalization) 
    {
        //A_n has r rows and c columns. We are going to make a new data matrix Z
        //Z will have (r+c) rows, and l columns.
        TruncatedSVD svd = new TruncatedSVD(A_n, l+1);//+1 b/c we are going to skip the first SV
        Matrix U = svd.getU();
        Matrix V = svd.getV().transpose();
        //In some cases, Drop the first column, which corresponds to the first SV we don't want
        int to_skip = 1;
        U = new SubMatrix(U, 0, to_skip, U.rows(), l+to_skip);
        V = new SubMatrix(V, 0, to_skip, V.rows(), l+to_skip);
        /* Orig paper says to do this multiplication for re-scaling. Why not for
         * bistochastic? Its very similar! b/c in "﻿Spectral Biclustering of 
         * Microarray Data: Coclustering Genes and Conditions" where bistochastic
         * is introduced, on page 710: "﻿Once D1 and D2 are found, we ﻿apply SVD to 
         * B with no further normalization "
         * 
         */
        if(inputNormalization == InputNormalization.SCALE)
        {
            Matrix.diagMult(R, U);
            Matrix.diagMult(C, V);
        }
        
        SimpleDataSet Z = new SimpleDataSet(l, new CategoricalData[0]);
        for(int i = 0; i < U.rows(); i++)
            Z.add(new DataPoint(U.getRow(i)));
        for(int i = 0; i < V.rows(); i++)
            Z.add(new DataPoint(V.getRow(i)));
        return Z;
    }
    
    private void createAssignments(List<List<Integer>> row_assignments, List<List<Integer>> col_assignments, int clusters, Matrix A, int[] joint_designations) {
        //prep label outputs
        row_assignments.clear();
        col_assignments.clear();
        for(int c = 0; c < clusters; c++)
        {
            row_assignments.add(new IntList());
            col_assignments.add(new IntList());
        }
        
        for(int i = 0; i < A.rows(); i++)//the bicluster labels for the rows
            if(joint_designations[i] >= 0)
                row_assignments.get(joint_designations[i]).add(i);
        for(int j = 0; j < A.cols(); j++)//the bicluter labels for the columns
            if(joint_designations[j+A.rows()] >= 0)
                col_assignments.get(joint_designations[j+A.rows()]).add(j);
        //Now we need to prune potential false bi-clusterings that have only features or only rows
        for(int j = row_assignments.size()-1; j >= 0; j--)
        {
            if(row_assignments.get(j).isEmpty() || col_assignments.get(j).isEmpty())
            {
                row_assignments.remove(j);
                col_assignments.remove(j);
            }
        }
    }


    @Override
    public SpectralCoClustering clone() 
    {
        return this;
    }
    
    /**
     * Performs normalization as described in Section 4.of "﻿Co-clustering
     * Documents and Words Using Bipartite Spectral Graph Partitioning"
     *
     * @param A the matrix to normalize
     * @param R the location to store the row sums, should have length equal to
     * the number of rows in A
     * @param C the location to store the column sums, should have length equal
     * to the number of columns in A.
     * @return a normalized copy of the original matrix.
     */
    protected static Matrix row_col_normalize(Matrix A, Vec R, Vec C) 
    {
        R.zeroOut();
        C.zeroOut();
        //A_n = R^{−1/2} A C^{−1/2}
        //Where R and C are diagonal matrix with Row and Column sums
        for (int i = 0; i < A.rows(); i++)
            for(IndexValue iv : A.getRowView(i))
            {
                int j = iv.getIndex();
                double v = iv.getValue();

                R.increment(i, v);
                C.increment(j, v);
            }
        
        R.applyFunction(v -> v == 0 ? 0 : 1.0/Math.sqrt(v));
        C.applyFunction(v -> v == 0 ? 0 : 1.0/Math.sqrt(v));
        
        Matrix A_n = A.clone();
        Matrix.diagMult(R, A_n);
        Matrix.diagMult(A_n, C);
        
        return A_n;
    }
    
}
