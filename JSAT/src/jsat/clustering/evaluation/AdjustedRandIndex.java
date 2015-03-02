package jsat.clustering.evaluation;

import java.util.List;
import jsat.DataSet;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.DataPoint;
import static java.lang.Math.*;

/**
 * Adjusted Rand Index (ARI) is a measure to evaluate a cluster based 
 * on the true class labels for the data set. The ARI normally returns a value 
 * in [-1, 1], where 0 indicates the clustering appears random, and 1 indicate 
 * the clusters perfectly match the class labels, and negative values indicate a
 * clustering that is worse than random. To match the 
 * {@link ClusterEvaluation} interface, the value returned by evaluate will 
 * be 1.0-Adjusted Rand Index so the best value becomes 0.0 and the worse value 
 * becomes 2.0. 
 * <br>
 * <b>NOTE:</b> Because the ARI needs to know the true class labels, only 
 * {@link #evaluate(int[], jsat.DataSet) } will work, since it provides the data
 * set as an argument. The dataset given must be an instance of 
 * {@link ClassificationDataSet}
 * 
 * @author Edward Raff
 */
public class AdjustedRandIndex implements ClusterEvaluation 
{

    @Override
    public double evaluate(int[] designations, DataSet dataSet)
    {
        if( !(dataSet instanceof ClassificationDataSet))
            throw new RuntimeException("NMI can only be calcuate for classification data sets");
        ClassificationDataSet cds = (ClassificationDataSet)dataSet;
        int clusters = 0;//how many clusters are there? 
        for(int clusterID : designations)
            clusters = Math.max(clusterID+1, clusters);
        double[] truthSums = new double[cds.getClassSize()];
        double[] clusterSums = new double[clusters];
        double[][] table = new double[clusterSums.length][truthSums.length];
        double n = 0.0;
        for(int i = 0; i < designations.length; i++)
        {
            int cluster = designations[i];
            if(cluster < 0)
                continue;//noisy point 
            int label = cds.getDataPointCategory(i);
            double weight = cds.getDataPoint(i).getWeight();
            table[cluster][label] += weight;
            truthSums[label] += weight;
            clusterSums[cluster] += weight;
            n += weight;
        }
        
        /*
        * Adjusted Rand Index involves many (n choose 2) = 1/2 (n-1) n
        */
        
        double sumAllTable = 0.0;
        double addCTerm = 0.0, addLTerm = 0.0;//clustering and label
        
        for(int i = 0; i < table.length; i++)
        {
            double a_i = clusterSums[i];
            addCTerm += a_i*(a_i-1)/2;
            
            for(int j = 0; j < table[i].length; j++)
            {
                if(i == 0)
                {
                    double b_j = truthSums[j];
                    addLTerm += b_j*(b_j-1)/2;
                }
                
                double n_ij = table[i][j];
                double n_ij_c2 = n_ij*(n_ij-1)/2;
                sumAllTable += n_ij_c2;
            }
        }
        
        double longMultTerm = exp(log(addCTerm)+log(addLTerm)-(log(n)+log(n-1)-log(2)));//numericaly more stable verison
        return 1.0-(sumAllTable-longMultTerm)/(addCTerm/2+addLTerm/2-longMultTerm);
    }

    @Override
    public double evaluate(List<List<DataPoint>> dataSets)
    {
        throw new UnsupportedOperationException("Adjusted Rand Index requires the true data set"
                + " labels, call evaluate(int[] designations, DataSet dataSet)"
                + " instead");
    }

    @Override
    public ClusterEvaluation clone()
    {
        return new AdjustedRandIndex();
    }
}
