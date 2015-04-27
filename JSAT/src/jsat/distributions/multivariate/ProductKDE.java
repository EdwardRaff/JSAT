
package jsat.distributions.multivariate;

import static java.lang.Math.*;

import java.util.*;

import jsat.classifiers.DataPoint;
import jsat.distributions.empirical.KernelDensityEstimator;
import jsat.distributions.empirical.kernelfunc.EpanechnikovKF;
import jsat.distributions.empirical.kernelfunc.KernelFunction;
import jsat.exceptions.UntrainedModelException;
import jsat.linear.*;
import jsat.utils.IndexTable;
import jsat.utils.IntSet;

/**
 * The Product Kernel Density Estimator is a generalization of the {@link KernelDensityEstimator} to the multivariate case. 
 * This is done by using a kernel and bandwidth for each dimension, such that the bandwidth for each dimension can be 
 * determined using the same methods as the univariate KDE.  This can simplify the difficulty in bandwidth selection 
 * for the multivariate case. 
 * 
 * @author Edward Raff
 * @see MetricKDE
 */
public class ProductKDE extends MultivariateKDE
{

	private static final long serialVersionUID = 7298078759216991650L;
	private KernelFunction k;
    private double[][] sortedDimVals;
    private double[] bandwidth;
    private int[][] sortedIndexVals;
    /**
     * The original list of vectors used to create the KDE, used to avoid an expensive reconstruction of the vectors 
     */
    private List<Vec> originalVecs;

    /**
     * Creates a new KDE that uses the {@link EpanechnikovKF} kernel. 
     */
    public ProductKDE()
    {
        this(EpanechnikovKF.getInstance());
    }
    
    /**
     * Creates a new KDE that uses the specified kernel
     * @param k the kernel method to use 
     */
    public ProductKDE(KernelFunction k)
    {
        this.k = k;
    }

    @Override
    public ProductKDE clone()
    {
        ProductKDE clone = new ProductKDE();
        if(this.k != null)
            clone.k = k;
        if(this.sortedDimVals != null)
        {
            clone.sortedDimVals = new double[sortedDimVals.length][];
            for(int i = 0; i < this.sortedDimVals.length; i++)
                clone.sortedDimVals[i] = Arrays.copyOf(this.sortedDimVals[i], this.sortedDimVals[i].length);
        }
        if(this.sortedIndexVals != null)
        {
            clone.sortedIndexVals = new int[sortedIndexVals.length][];
            for(int i = 0; i < this.sortedIndexVals.length; i++)
                clone.sortedIndexVals[i] = Arrays.copyOf(this.sortedIndexVals[i], this.sortedIndexVals[i].length);
        }
        if(this.bandwidth != null)
            clone.bandwidth = Arrays.copyOf(this.bandwidth, this.bandwidth.length);
        if(this.originalVecs != null)
            clone.originalVecs = new ArrayList<Vec>(this.originalVecs);
        return clone;        
    }

    @Override
    public List<VecPaired<VecPaired<Vec, Integer>, Double>> getNearby(Vec x)
    {
        
        SparseVector logProd = new SparseVector(sortedDimVals[0].length);
        Set<Integer> validIndecies = new IntSet();
        double logH = queryWork(x, validIndecies, logProd);
        List<VecPaired<VecPaired<Vec, Integer>, Double>> results = new ArrayList<VecPaired<VecPaired<Vec, Integer>, Double>>(validIndecies.size());
        
        for(int i : validIndecies)
        {
            Vec v = originalVecs.get(i);
            results.add(new VecPaired<VecPaired<Vec, Integer>, Double>(new VecPaired<Vec, Integer>(v, i), exp(logProd.get(i))));
        }
        return results;
    }
    
    @Override
    public List<VecPaired<VecPaired<Vec, Integer>, Double>> getNearbyRaw(Vec x)
    {
        //Not entirly sure how I'm going to fix this... but this isnt technically right
        throw new UnsupportedOperationException("Product KDE can not recover raw Score values");
    }
    
    @Override
    public double pdf(Vec x)
    {
        double PDF = 0;
        int N = sortedDimVals[0].length;
        
        SparseVector logProd = new SparseVector(sortedDimVals[0].length);
        Set<Integer> validIndecies = new IntSet();
        double logH = queryWork(x, validIndecies, logProd);
        
        for(int i : validIndecies)
            PDF += exp(logProd.get(i)-logH);
        
        return PDF/N;
    }

    /**
     * Performs the main work for performing a density query. 
     * 
     * @param x the query vector 
     * @param validIndecies the empty set that will be altered to contain the 
     * indices of vectors that had a non zero contribution to the density 
     * @param logProd an empty sparce vector that will be modified to contain the log of the product of the 
     * kernels for each data point. Some indices that have zero contribution to the density will have non 
     * zero values. <tt>validIndecies</tt> should be used to access the correct indices. 
     * @return The log product of the bandwidths that normalizes the values stored in the <tt>logProd</tt> vector. 
     */
    private double queryWork(Vec x, Set<Integer> validIndecies, SparseVector logProd)
    {
        if(originalVecs == null)
            throw new UntrainedModelException("Model has not yet been created, queries can not be perfomed");
        double logH = 0;
        for(int i = 0; i < sortedDimVals.length; i++)
        {
            double[] X = sortedDimVals[i];
            double h = bandwidth[i];
            logH += log(h);
            double xi = x.get(i);

            //Only values within a certain range will have an effect on the result, so we will skip to that range!
            int from = Arrays.binarySearch(X, xi-h*k.cutOff());
            int to = Arrays.binarySearch(X, xi+h*k.cutOff());
            //Mostly likely the exact value of x is not in the list, so it retursn the inseration points
            from = from < 0 ? -from-1 : from;
            to = to < 0 ? -to-1 : to;
            Set<Integer> subIndecies = new IntSet();
            for(int j = max(0, from); j < min(X.length, to+1); j++)
            {
                int trueIndex = sortedIndexVals[i][j];
                
                if(i == 0)
                {
                    validIndecies.add(trueIndex);
                    logProd.set(trueIndex, log(k.k( (xi-X[j])/h )));
                }
                else if(validIndecies.contains(trueIndex))
                {
                    logProd.increment(trueIndex, log(k.k( (xi-X[j])/h )));
                    subIndecies.add(trueIndex);
                }
            }

            if (i > 0)
            {
                validIndecies.retainAll(subIndecies);
                if(validIndecies.isEmpty())
                    break;
            }
        }
        return logH;
    }

    @Override
    public <V extends Vec> boolean setUsingData(List<V> dataSet)
    {
        int dimSize = dataSet.get(0).length();
        sortedDimVals = new double[dimSize][dataSet.size()];
        sortedIndexVals = new int[dimSize][dataSet.size()];
        bandwidth = new double[dimSize];
        
        for(int i = 0; i < dataSet.size(); i++)
        {
            Vec v = dataSet.get(i);
            for(int j = 0; j < v.length(); j++)
                sortedDimVals[j][i] = v.get(j);
        }
        
        
        for(int i = 0; i < dimSize; i++)
        {
            IndexTable idt = new IndexTable(sortedDimVals[i]);
            for( int j = 0; j < idt.length(); j++)
                sortedIndexVals[i][j] = idt.index(j);
            idt.apply(sortedDimVals[i]);
            bandwidth[i] = KernelDensityEstimator.BandwithGuassEstimate(DenseVector.toDenseVec(sortedDimVals[i]))*dimSize;
        }
        this.originalVecs = (List<Vec>) dataSet;
        
        return true;
    }

    @Override
    public boolean setUsingDataList(List<DataPoint> dataPoints)
    {
        List<Vec> dataSet = new ArrayList<Vec>(dataPoints.size());
        for(DataPoint dp : dataPoints)
            dataSet.add(dp.getNumericalValues());
        return setUsingData(dataSet);
    }

    @Override
    public List<Vec> sample(int count, Random rand)
    {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    @Override
    public KernelFunction getKernelFunction()
    {
        return k;
    }

    @Override
    public void scaleBandwidth(double scale)
    {
        for(int i = 0; i < bandwidth.length; i++)
            bandwidth[i] *= 2;
    }
}
