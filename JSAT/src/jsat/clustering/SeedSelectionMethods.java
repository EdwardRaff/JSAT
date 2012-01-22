
package jsat.clustering;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import jsat.DataSet;
import jsat.linear.Vec;
import jsat.linear.distancemetrics.DistanceMetric;
import jsat.linear.distancemetrics.EuclideanDistance;

/**
 * This class provides methods for sampling a data set for a set of initial points to act as the seeds for a clustering algorithm. 
 * 
 * @author Edward Raff
 */
public class SeedSelectionMethods
{
    static public enum SeedSelection 
    {
        /**
         * The seed values will be randomly selected from the data set
         */
        RANDOM, 
        
        /**
         * The k-means++ seeding algo: <br>
         * The seed values will be probabilistically selected from the 
         * data set. <br>
         * The solution is O(log(k)) competitive with the 
         * optimal k clustering when using {@link EuclideanDistance}. 
         * <br><br>
         * See k-means++: The Advantages of Careful Seeding
         */
        KPP
    };
    
    /**
     * Selects seeds from a data set to use for a clustering algorithm. Copies of the vectors chosen will be returned.
     * 
     * @param d the data set to perform select from
     * @param k the number of seeds to choose 
     * @param dm the distance metric to used when selecting points
     * @param rand a source of randomness
     * @param selectionMethod  The method of seed selection to use. 
     * @return a list of the copies of the chosen vectors. 
     */
    static protected List<Vec> selectIntialPoints(DataSet d, int k, DistanceMetric dm, Random rand, SeedSelection selectionMethod)
    {
        int[] indicies = new int[k];
        selectIntialPoints(d, indicies, dm, rand, selectionMethod);
        List<Vec> vecs = new ArrayList<Vec>(k);
        for(Integer i : indicies)
            vecs.add(d.getDataPoint(i).getNumericalValues().clone());
        return vecs;
    }
    /**
     * Selects seeds from a data set to use for a clustering algorithm. The indices of the chosen points will be placed in the <tt>indices</tt> array. 
     * 
     * @param d the data set to perform select from
     * @param indices a storage place to note the indices that were chosen as seed. The length of the array indicates how many seeds to select. 
     * @param dm the distance metric to used when selecting points
     * @param rand a source of randomness
     * @param selectionMethod  The method of seed selection to use. 
     */
    static protected void selectIntialPoints(DataSet d, int[] indices, DistanceMetric dm, Random rand, SeedSelection selectionMethod)
    {
        int k = indices.length;
        
        if(selectionMethod ==  SeedSelection.RANDOM)
        {
            Set<Integer> indecies = new HashSet<Integer>(k);

            while(indecies.size() != k)//Keep sampling, we cant use the same point twice. 
                indecies.add(rand.nextInt(d.getSampleSize()));//TODO create method to do uniform sampleling for a select range

            int j = 0;
            for(Integer i : indecies)
                indices[j++] = i;
        }
        else if(selectionMethod == SeedSelection.KPP)
        {
            /*
             * http://www.stanford.edu/~darthur/kMeansPlusPlus.pdf : k-means++: The Advantages of Careful Seeding
             * 
             */
            //Initial random point
            indices[0] = rand.nextInt(d.getSampleSize());
            
            double[] closestDist = new double[d.getSampleSize()];
            double sqrdDistSum = 0.0;
            double newDist;
            
            for(int j = 1; j < k; j++)
            {
                newDist = Double.MAX_VALUE;
                
                //Compute the distance from each data point to the closest mean
                Vec newMean = d.getDataPoint(indices[j-1]).getNumericalValues();//Only the most recently added mean needs to get distances computed. 
                for(int i = 0; i < d.getSampleSize(); i++)
                {
                    newDist = dm.dist(newMean, d.getDataPoint(i).getNumericalValues());
                    
                    if(newDist < closestDist[i] || j == 1)
                    {
                        newDist*=newDist;
                        sqrdDistSum -= closestDist[i];//on inital, -= 0  changes nothing. on others, removed the old value
                        sqrdDistSum += newDist;
                        closestDist[i] = newDist;
                    }
                }
                
                //Choose new x as weighted probablity by the squared distances
                double rndX = rand.nextDouble()*sqrdDistSum;
                double searchSum = 0;
                int i = -1;
                while(searchSum < rndX && i < d.getSampleSize()-1)
                    searchSum += closestDist[++i];
                
                indices[j] = i;
            }
        }
    }
    
}
