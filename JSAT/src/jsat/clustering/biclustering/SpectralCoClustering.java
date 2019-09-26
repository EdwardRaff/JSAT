/*
 * This code was contributed under the public domain. 
 */
package jsat.clustering.biclustering;

import java.util.List;
import jsat.DataSet;
import jsat.SimpleDataSet;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.DataPoint;
import jsat.clustering.KClusterer;
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
 * @author edwardraff
 */
public class SpectralCoClustering implements Bicluster
{

    @Override
    public void bicluster(DataSet dataSet, int clusters, boolean parallel, List<List<Integer>> row_assignments, List<List<Integer>> col_assignments) 
    {
        //﻿1. Given A, form An = D_1^{−1/2} A D_2^{−1/2}
        Matrix A = dataSet.getDataMatrix();
                        
        DenseVector R = new DenseVector(A.rows());
        DenseVector C = new DenseVector(A.cols());
        
        Matrix A_n = row_col_normalize(A, R, C);
        
        //﻿2. Compute l = ceil(log2 k) singular vectors of A_n, u2, . . . u_l+1 and v2, . . . v_l+1, and form the matrix Z as in (12)
        int l = (int) Math.ceil(Math.log(clusters)/Math.log(2.0));
        
        
        //A_n has r rows and c columns. We are going to make a new data matrix Z
        //Z will have (r+c) rows, and l columns. 
        TruncatedSVD svd = new TruncatedSVD(A_n, l+1);//+1 b/c we are going to skip the first SV
        Matrix U = svd.getU();
        Matrix V = svd.getV().transpose();
        
        //Drop the first column, which corresponds to the first SV we don't want
        U = new SubMatrix(U, 0, 1, U.rows(), l+1);
        V = new SubMatrix(V, 0, 1, V.rows(), l+1);
        
        Matrix.diagMult(R, U);
        Matrix.diagMult(C, V);
        
        
        SimpleDataSet Z = new SimpleDataSet(l, new CategoricalData[0]);
        for(int i = 0; i < U.rows(); i++)
            Z.add(new DataPoint(U.getRow(i)));
        for(int i = 0; i < V.rows(); i++)
            Z.add(new DataPoint(V.getRow(i)));
        
        KClusterer kMeans = new HamerlyKMeans();
        int[] joint_designations = kMeans.cluster(Z, clusters, parallel, null);
        
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
