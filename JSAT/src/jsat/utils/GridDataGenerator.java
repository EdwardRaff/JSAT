
package jsat.utils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import jsat.SimpleDataSet;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.DataPoint;
import jsat.distributions.ContinuousDistribution;
import jsat.distributions.Uniform;
import jsat.linear.DenseVector;
import jsat.utils.random.RandomUtil;

/**
 * This is a utility to generate data in a grid fashion. 
 * Each data point will belong to a different class. By default, 
 * the data is generated such that the classes are trivially separable. 
 * <br><br>
 * All data points belong to a specific axis, and have noises added to them. 
 * For example, a 1 dimensional grid with 2 classes would have data points 
 * of the form 0+noise, and 1+noise. So long as the noise is less than 0.5,
 * the data can be easily separated. 
 * 
 * @author Edward Raff
 */
public class GridDataGenerator
{
   private ContinuousDistribution noiseSource; 
   private int[] dimensions;
   private Random rand;
   private CategoricalData[] catDataInfo;

   /**
    * Creates a new Grid data generator, that can be queried for new data sets. 
    * 
    * @param noiseSource the distribution that describes the noise that will be added to each data point. If no noise is used, each data point would lie exactly on its local axis.  
    * @param rand the source of randomness that the noise will use
    * @param dimensions an array describing how many groups there will be. 
    * The length of the array dictates the number of dimensions in the data
    * set, each value describes how many axis of that dimensions to use. The
    * total number of classes is the product of these values. 
    * 
    * @throws ArithmeticException if one of the dimension values is not a positive value, or a zero number of dimensions is given
    */
    public GridDataGenerator(ContinuousDistribution noiseSource, Random rand, int... dimensions)
    {
        this.noiseSource = noiseSource;
        this.rand = rand;
        this.dimensions = dimensions;
        for(int i = 0; i < dimensions.length; i++)
            if(dimensions[i] <= 0)
                throw new ArithmeticException("The " + i + "'th dimensino contains the non positive value " + dimensions[i]);
    }

    /**
     * Creates a new Grid data generator, that can be queried for new data sets. 
     * 
     * @param noiseSource  the distribution that describes the noise that will be added to each data point. If no noise is used, each data point would lie exactly on its local axis.  
     * @param dimensions an array describing how many groups there will be. 
     * The length of the array dictates the number of dimensions in the data
     * set, each value describes how many axis of that dimensions to use. The
     * total number of classes is the product of these values. 
     * 
     * @throws ArithmeticException if one of the dimension values is not a positive value, or a zero number of dimensions is given
     */
    public GridDataGenerator(ContinuousDistribution noiseSource, int... dimensions)
    {
        this(noiseSource, RandomUtil.getRandom(), dimensions);
    }

    /**
     * Creates a new grid data generator for a 2 x 5 with uniform noise in the range [-1/4, 1/4]
     */
    public GridDataGenerator()
    {
        this(new Uniform(-0.25, 0.25), RandomUtil.getRandom(), 2, 5);
    }
    
    
    /**
     * Helper function
     * @param curClass used as a pointer to an integer so that we dont have to add class tracking logic
     * @param curDim the current dimension to split on. If we are at the last dimension, we add data points instead. 
     * @param samples the number of samples to take for each class
     * @param dataPoints the location to put the data points in 
     * @param dim the array specifying the current point we are working from. 
     */
    private void addSamples(int[] curClass, int curDim, int samples, List<DataPoint> dataPoints, int[] dim)
    {
        if(curDim < dimensions.length-1)
            for(int i = 0; i < dimensions[curDim+1]; i++ )
            {
                int[] nextDim = Arrays.copyOf(dim, dim.length);
                nextDim[curDim+1] = i;
                addSamples(curClass, curDim+1, samples, dataPoints, nextDim);
            }
        else//Add data points!
        {
            for(int i = 0; i < samples; i++)
            {
                DenseVector dv = new DenseVector(dim.length);
                for(int j = 0; j < dim.length; j++)
                    dv.set(j, dim[j]+noiseSource.invCdf(rand.nextDouble()));
                dataPoints.add(new DataPoint(dv, new int[]{ curClass[0] }, catDataInfo));
            }
            curClass[0]++;
        }
    }
    
    /**
     * Generates a new data set. 
     * 
     * @param samples the number of sample data points to create for each class in the data set. 
     * @return A data set the contains the data points with matching class labels. 
     */
    public SimpleDataSet generateData(int samples)
    {
        int totalClasses = 1;
        for(int d : dimensions)
            totalClasses *= d;
        catDataInfo = new CategoricalData[] { new CategoricalData(totalClasses) } ;
        List<DataPoint> dataPoints = new ArrayList<DataPoint>(totalClasses*samples);
        int[] curClassPointer = new int[1];
                
        for(int i = 0; i < dimensions[0]; i++)
        {
            int[] curDim = new int[dimensions.length];
            curDim[0] = i;
            addSamples(curClassPointer, 0, samples, dataPoints, curDim);
        }
        
        return new SimpleDataSet(dataPoints);
    }
    
}
