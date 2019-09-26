/*
 * This code was contributed under the public domain. 
 */
package jsat.clustering.biclustering;

import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import jsat.linear.DenseMatrix;
import jsat.linear.Matrix;
import jsat.utils.Pair;
import jsat.utils.concurrent.ParallelUtils;

/**
 * Implementatino of the Consensus Score method for evaluating the quality of a
 * biclustering compared to a known ground truth.<br>
 * <br>
 * See: ﻿[1] S. Hochreiter et al., “<i>FABIA: factor analysis for bicluster acquisition</i>,” Bioinformatics, vol. 26, no. 12, pp. 1520–1527, Jun. 2010.
 * @author Edward Raff
 */
public class ConsensusScore 
{
    /**
     * Computes the Consensus score to measure the quality of a biclustering
     * algorithm.
     *
     * @param parallel whether or not multiple threads should be used in the
     * computation of this score.
     * @param rows_truth A list for each bicluster, where the i'th sub-list
     * contains the rows of the i'th biclustering. This biclustering should
     * correspond to the ground-truth that is known in advance.
     * @param cols_truth A list for each bicluster, where the i'th sub-list
     * contains the columns of the i'th biclustering. This biclustering should
     * correspond to the ground-truth that is known in advance.
     * @param rows_found A list for each bicluster, where the i'th sub-list
     * contains the rows of the i'th biclustering. This biclustering should
     * correspond to the found clustering that we wish to evaluate.
     * @param cols_found A list for each bicluster, where the i'th sub-list
     * contains the columns of the i'th biclustering. This biclustering should
     * correspond to the found clustering that we wish to evaluate.
     * @return a score in the range of [0, 1]. Where 1 indicate a perfect
     * biclustering, and 0 indicates a biclustering with no overlap of the
     * ground truth.
     */
    public static double score(boolean parallel, 
            List<List<Integer>> rows_truth, List<List<Integer>> cols_truth,
            List<List<Integer>> rows_found, List<List<Integer>> cols_found)
    {
        int k_true = rows_truth.size();
        int k_found = rows_found.size();
        
        //﻿(1) compute similarities between all pairs of biclusters, where one is 
        //from the first set and the other from the second set;
        double[][] cost_matrix = new double[k_true][k_found];
        ParallelUtils.run(parallel, k_true, (i)->
        {
            Set<Pair<Integer, Integer>> true_ci = coCluster_to_set(rows_truth, i, cols_truth);
            for(int j = 0; j < k_found; j++)
            {
                Set<Pair<Integer, Integer>> true_cj = coCluster_to_set(rows_found, j, cols_found);
                int A_size = true_ci.size();
                int B_size = true_cj.size();
                
                //remove everything we don't have, so now we represent the union
                true_cj.removeIf(pair-> !true_ci.contains(pair));
                int union = true_cj.size();
                
                cost_matrix[i][j] = 1.0-union/(double)(A_size+B_size-union);
            }
        });
        
        
        //﻿(2) assign the biclusters of one set to biclusters of the other set by 
        //maximizing the assignment by the Munkres algorithm (Munkres, 1957); 
        Map<Integer, Integer> assignments = assignment(new DenseMatrix(cost_matrix));
        
        double score_sum = 0;
        
        //﻿(3) divide the sum of similarities of the assigned biclusters ...
        for(Map.Entry<Integer, Integer> pair : assignments.entrySet())
            score_sum += (1.0-cost_matrix[pair.getKey()][pair.getValue()]);
        //by the number of biclusters of the larger set
        return score_sum/Math.max(k_true, k_found);
    }

    private static Set<Pair<Integer, Integer>> coCluster_to_set(List<List<Integer>> rows_truth, int q, List<List<Integer>> cols_truth) {
        Set<Pair<Integer, Integer>> true_c_i = new HashSet<>();
        List<Integer> rows = rows_truth.get(q);
        List<Integer> cols = cols_truth.get(q);
        for(int i  = 0; i < rows.size(); i++)
        {
            for(int j  = 0; j < cols.size(); j++)
                true_c_i.add(new Pair<>(rows.get(i), cols.get(j)));
        }
        
        return true_c_i;
    }
    private static Map<Integer, Integer> assignment(Matrix A)
    {
        Map<Integer, Integer> assignments = new HashMap<>();
        boolean[] taken = new boolean[A.cols()];
        
        //TODO, greedy assignment that is not optimal. Replace with hungarian or something
        int min_indx;
        double best_score;
        for(int i = 0; i < A.rows(); i++)
        {
            min_indx = -1;
            best_score = Double.POSITIVE_INFINITY;
            for(int j = 0; j < A.cols(); j++)
            {
                double score = A.get(i, j);
                if(score < best_score && !taken[j])
                {
                    best_score = score;
                    min_indx = j;
                }
            }
            
            assignments.put(i, min_indx);
            taken[min_indx] = true;
            
            if(assignments.size() == Math.min(A.rows(), A.cols()))
                break;//Nothing left to match up
        }
        
        return assignments;
    }
}
