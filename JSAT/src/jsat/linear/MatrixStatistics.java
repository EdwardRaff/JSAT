
package jsat.linear;

import static java.lang.Math.pow;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Random;
import java.util.Set;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import jsat.DataSet;
import jsat.classifiers.DataPoint;
import jsat.distributions.ChiSquared;
import jsat.linear.distancemetrics.MahalanobisDistance;
import jsat.utils.IndexTable;
import jsat.utils.IntList;
import jsat.utils.IntSet;
import jsat.utils.ListUtils;
import jsat.utils.Tuple3;
import jsat.utils.concurrent.ParallelUtils;
import jsat.utils.random.RandomUtil;

/**
 * This class provides methods useful for statistical operations that involve matrices and vectors. 
 * 
 * @author Edward Raff
 */
public class MatrixStatistics
{
    private MatrixStatistics()
    {
        
    }
    
    /**
     * Computes the mean of the given data set.  
     * @param <V> the vector type
     * @param dataSet the list of vectors to compute the mean of
     * @return the mean of the vectors 
     */
    public static <V extends Vec> Vec meanVector(List<V> dataSet)
    {
        if(dataSet.isEmpty())
            throw new ArithmeticException("Can not compute the mean of zero data points");
        
        Vec mean = new DenseVector(dataSet.get(0).length());
        meanVector(mean, dataSet);
        return mean;
    }
    
    /**
     * Computes the weighted mean of the given data set. 
     * 
     * @param dataSet the dataset to compute the mean from
     * @return the mean of the numeric vectors in the data set
     */
    public static Vec meanVector(DataSet dataSet)
    {
        DenseVector dv = new DenseVector(dataSet.getNumNumericalVars());
        meanVector(dv, dataSet);
        return dv;
    }
    
    /**
     * Computes the mean of the given data set. 
     * 
     * @param mean the zeroed out vector to store the mean in. Its contents will be altered
     * @param dataSet the set of data points to compute the mean from
     */
    public static <V extends Vec> void meanVector(Vec mean, List<V> dataSet)
    {
        if(dataSet.isEmpty())
            throw new ArithmeticException("Can not compute the mean of zero data points");
        else if(dataSet.get(0).length() != mean.length())
            throw new ArithmeticException("Vector dimensions do not agree");

        for (Vec x : dataSet)
            mean.mutableAdd(x);
        mean.mutableDivide(dataSet.size());
    }
    
    /**
     * Computes the mean of the given data set. 
     * 
     * @param <V>
     * @param mean the zeroed out vector to store the mean in. Its contents will be altered
     * @param dataSet the set of data points to compute the mean from
     * @param subset the indecies of the points in dataSet to take the mean of
     */
    public static <V extends Vec> void meanVector(Vec mean, List<V> dataSet, Collection<Integer> subset)
    {
        if(dataSet.isEmpty())
            throw new ArithmeticException("Can not compute the mean of zero data points");
        else if(dataSet.get(0).length() != mean.length())
            throw new ArithmeticException("Vector dimensions do not agree");

        for(int i : subset)
            mean.mutableAdd(dataSet.get(i));
        mean.mutableDivide(subset.size());
    }
    
    /**
     * Computes the weighted mean of the data set
     * @param mean the zeroed out vector to store the mean in. Its contents will be altered
     * @param dataSet the set of data points to compute the mean from
     */
    public static void meanVector(Vec mean, DataSet dataSet)
    {
        if(dataSet.size() == 0)
            throw new ArithmeticException("Can not compute the mean of zero data points");
        double sumOfWeights = 0;
        for(int i = 0; i < dataSet.size(); i++)
        {
            DataPoint dp = dataSet.getDataPoint(i);
            double w = dataSet.getWeight(i);
            sumOfWeights += w;
            mean.mutableAdd(w, dp.getNumericalValues());
        }
        mean.mutableDivide(sumOfWeights);
    }
    
    public static <V extends Vec> Matrix covarianceMatrix(Vec mean, List<V> dataSet)
    {
        Matrix coMatrix = new DenseMatrix(mean.length(), mean.length());
        covarianceMatrix(mean, coMatrix, dataSet);
        return coMatrix;
    }
    
    public static <V extends Vec> void covarianceMatrix(Vec mean, Matrix covariance, List<V> dataSet)
    {
        if(!covariance.isSquare())
            throw new ArithmeticException("Storage for covariance matrix must be square");
        else if(covariance.rows() != mean.length())
            throw new ArithmeticException("Covariance Matrix size and mean size do not agree");
        else if(dataSet.isEmpty())
            throw new ArithmeticException("No data points to compute covariance from");
        else if(mean.length() != dataSet.get(0).length())
            throw new ArithmeticException("Data vectors do not agree with mean and covariance matrix");
        /**
         * Covariance definition
         * 
         *   n
         * =====                    T 
         * \     /     _\  /     _\
         *  >    |x  - x|  |x  - x|
         * /     \ i    /  \ i    /
         * =====
         * i = 1
         * 
         */
        Vec scratch = new DenseVector(mean.length());
        for (Vec x : dataSet)
        {
            x.copyTo(scratch);
            scratch.mutableSubtract(mean);
            Matrix.OuterProductUpdate(covariance, scratch, scratch, 1.0);
        }
        covariance.mutableMultiply(1.0 / (dataSet.size() - 1.0));
    }
    
    public static <V extends Vec> void covarianceMatrix(Vec mean, Matrix covariance, List<V> dataSet, Collection<Integer> subset)
    {
        if(!covariance.isSquare())
            throw new ArithmeticException("Storage for covariance matrix must be square");
        else if(covariance.rows() != mean.length())
            throw new ArithmeticException("Covariance Matrix size and mean size do not agree");
        else if(dataSet.isEmpty())
            throw new ArithmeticException("No data points to compute covariance from");
        else if(mean.length() != dataSet.get(0).length())
            throw new ArithmeticException("Data vectors do not agree with mean and covariance matrix");
        /**
         * Covariance definition
         * 
         *   n
         * =====                    T 
         * \     /     _\  /     _\
         *  >    |x  - x|  |x  - x|
         * /     \ i    /  \ i    /
         * =====
         * i = 1
         * 
         */
        Vec scratch = new DenseVector(mean.length());
        for(int i : subset)
        {
            dataSet.get(i).copyTo(scratch);
            scratch.mutableSubtract(mean);
            Matrix.OuterProductUpdate(covariance, scratch, scratch, 1.0);
        }
        covariance.mutableMultiply(1.0 / (subset.size() - 1.0));
    }
    
    /**
     * Computes the weighted result for the covariance matrix of the given data set. 
     * If all weights have the same value, the result will come out equivalent to 
     * {@link #covarianceMatrix(jsat.linear.Vec, java.util.List) }
     * 
     * @param mean the mean of the distribution. 
     * @param dataSet the set of data points that contain vectors 
     * @param covariance the zeroed matrix to store the result in. Its values will be altered.
     */
    public static void covarianceMatrix(Vec mean, DataSet dataSet, Matrix covariance)
    {
        double sumOfWeights = 0.0, sumOfSquaredWeights = 0.0;
        
        for(int i = 0; i < dataSet.size(); i++)
        {
            sumOfWeights += dataSet.getWeight(i);
            sumOfSquaredWeights += Math.pow(dataSet.getWeight(i), 2);
        }
        
        covarianceMatrix(mean, dataSet, covariance, sumOfWeights, sumOfSquaredWeights);
    }
    
    /**
     * Computes the weighted result for the covariance matrix of the given data set. 
     * If all weights have the same value, the result will come out equivalent to 
     * {@link #covarianceMatrix(jsat.linear.Vec, java.util.List) }
     * 
     * @param mean the mean of the distribution. 
     * @param dataSet the set of data points that contain vectors 
     * @param covariance the zeroed matrix to store the result in. Its values will be altered.
     * @param sumOfWeights the sum of each weight in <tt>dataSet</tt>
     * @param sumOfSquaredWeights  the sum of the squared weights in <tt>dataSet</tt>
     */
    public static void covarianceMatrix(Vec mean, DataSet dataSet, Matrix covariance, double sumOfWeights, double sumOfSquaredWeights)
    {
        if (!covariance.isSquare())
            throw new ArithmeticException("Storage for covariance matrix must be square");
        else if (covariance.rows() != mean.length())
            throw new ArithmeticException("Covariance Matrix size and mean size do not agree");
        else if (dataSet.isEmpty())
            throw new ArithmeticException("No data points to compute covariance from");
        else if (mean.length() != dataSet.getNumNumericalVars())
            throw new ArithmeticException("Data vectors do not agree with mean and covariance matrix");

        /**
         * Weighted definition of the covariance matrix 
         * 
         *          n
         *        =====
         *        \
         *         >    w
         *        /      i          n
         *        =====           =====                      T
         *        i = 1           \        /     _\  /     _\
         * ----------------------  >    w  |x  - x|  |x  - x|
         *           2            /      i \ i    /  \ i    /
         * /  n     \      n      =====
         * |=====   |    =====    i = 1
         * |\       |    \      2
         * | >    w |  -  >    w
         * |/      i|    /      i
         * |=====   |    =====
         * \i = 1   /    i = 1
         */

        Vec scratch = new DenseVector(mean.length());

        for (int i = 0; i < dataSet.size(); i++)
        {
            DataPoint dp = dataSet.getDataPoint(i);
            Vec x = dp.getNumericalValues();
            x.copyTo(scratch);
            scratch.mutableSubtract(mean);
            Matrix.OuterProductUpdate(covariance, scratch, scratch, dataSet.getWeight(i));
        }
        covariance.mutableMultiply(sumOfWeights / (Math.pow(sumOfWeights, 2) - sumOfSquaredWeights));
    }
    
    /**
     * Computes the weighted covariance matrix of the data set
     * @param mean the mean of the data set
     * @param dataSet the dataset to compute the covariance of
     * @return the covariance matrix of the data set
     */
    public static Matrix covarianceMatrix(Vec mean, DataSet dataSet)
    {
        Matrix covariance = new DenseMatrix(mean.length(), mean.length());
        covarianceMatrix(mean, dataSet, covariance);
        return covariance;
    }
    
    /**
     * Computes the weighted diagonal of the covariance matrix, which is the 
     * standard deviations of the columns of all values. 
     * 
     * @param means the already computed mean of the data set
     * @param diag the zeroed out vector to store the diagonal in. Its contents 
     * will be altered
     * @param dataset the data set to compute the covariance diagonal from
     */
    public static void covarianceDiag(Vec means, Vec diag, DataSet dataset)
    {
        final int n = dataset.size();
        final int d = dataset.getNumNumericalVars();
        
        int[] nnzCounts = new int[d];
        double sumOfWeights = 0;
        for(int i = 0; i < n; i++)
        {
            DataPoint dp = dataset.getDataPoint(i);
            double w = dataset.getWeight(i);
            sumOfWeights += w;
            Vec x = dataset.getDataPoint(i).getNumericalValues();
            for(IndexValue iv : x)
            {
                int indx = iv.getIndex();
                nnzCounts[indx]++;
                diag.increment(indx, w*pow(iv.getValue()-means.get(indx), 2));
            }
        }
        
        //add zero observations
        for(int i = 0; i < nnzCounts.length; i++)
            diag.increment(i, pow(means.get(i), 2)*(n-nnzCounts[i]) );
        diag.mutableDivide(sumOfWeights);
    }
    
    /**
     * Computes the weighted diagonal of the covariance matrix, which is the 
     * standard deviations of the columns of all values. 
     * 
     * @param means the already computed mean of the data set
     * @param dataset the data set to compute the covariance diagonal from
     * @return the diagonal of the covariance matrix for the given data 
     */
    public static Vec covarianceDiag(Vec means, DataSet dataset)
    {
        DenseVector diag = new DenseVector(dataset.getNumNumericalVars());
        covarianceDiag(means, diag, dataset);
        return diag;
    }
    
    /**
     * Computes the diagonal of the covariance matrix, which is the standard 
     * deviations of the columns of all values. 
     * 
     * @param <V> the type of the vector
     * @param means the already computed mean of the data set
     * @param diag the zeroed out vector to store the diagonal in. Its contents 
     * will be altered
     * @param dataset the data set to compute the covariance diagonal from
     */
    public static <V extends Vec> void covarianceDiag(Vec means, Vec diag, List<V> dataset)
    {
        final int n = dataset.size();
        final int d = dataset.get(0).length();
        
        int[] nnzCounts = new int[d];
        for(int i = 0; i < n; i++)
        {
            Vec x = dataset.get(i);
            for(IndexValue iv : x)
            {
                int indx = iv.getIndex();
                nnzCounts[indx]++;
                diag.increment(indx, pow(iv.getValue()-means.get(indx), 2));
            }
        }
        
        //add zero observations
        for(int i = 0; i < nnzCounts.length; i++)
            diag.increment(i, pow(means.get(i), 2)*(n-nnzCounts[i]) );
        diag.mutableDivide(n);
    }
    
    /**
     * Computes the diagonal of the covariance matrix, which is the standard 
     * deviations of the columns of all values. 
     * 
     * @param <V>
     * @param means the already computed mean of the data set
     * @param dataset the data set to compute the covariance diagonal from
     * @return the diagonal of the covariance matrix for the given data 
     */
    public static <V extends Vec> Vec covarianceDiag(Vec means, List<V> dataset)
    {
        final int d = dataset.get(0).length();
        DenseVector diag = new DenseVector(d);
        covarianceDiag(means, diag, dataset);;
        return diag;
    }
    
    /**
     * This algorithm implements the FastMCD algorithm for robustly estimating
     * the mean and covariance of a dataset. Computational complexity increases
     * linearly with the sample size {@code n}, but cubically with the dimension
     * size {@code d}.<br>
     * <br>
     * See: Rousseeuw, P. J., & Driessen, K. Van. (1999). A Fast Algorithm for
     * the Minimum Covariance Determinant Estimator. Technometrics, 41(3),
     * 212–223. http://doi.org/10.2307/1270566
     *
     * @param <V>
     * @param mean the location to store the estimated mean, values will be
     * overwritten
     * @param cov the location to store the estimated covariance, values will be
     * overwritten
     * @param dataset the set of data points to estimate the mean and covariance
     * of
     * @param parallel {@code true} if multiple cores should be used for
     * estimation, {@code false} for single thread.
     */
    public static  <V extends Vec> void FastMCD(Vec mean, Matrix cov, List<V> dataset, boolean parallel)
    {
        final int N = dataset.size();
        final int D = dataset.get(0).length();
        final int h = (int) Math.ceil((N + D + 1) / 2.0);
        
        
        mean.zeroOut();
        cov.zeroOut();
        
        if(h == N)
        {
            /*
             * 2. If h, = n,, then the MCD location estimate T is the average of
             * the whole dataset, and the MCD scatter estimate S is its 
             * covariance matrix. Report these and stop
             */
            meanVector(mean, dataset);
            covarianceMatrix(mean, cov, dataset);
            return;
        }
 
                    
        //Best results to store
        double bestDet = Double.POSITIVE_INFINITY;
        Vec bestMean = null;
        Matrix bestCov = null;
        
        if(N <= 600)
        {
            List<Tuple3<Double, Vec, Matrix>> top10 = 
                    ParallelUtils.range(500, parallel)
            .mapToObj(seed ->
            {
                Random rand = RandomUtil.getRandom(seed);
                
                Vec subset_mean = mean.clone();
                Matrix subset_cov = cov.clone();

                IntList randOrder = ListUtils.range(0, N);
                Collections.shuffle(randOrder, rand);

                IntList h_prev = new IntList( randOrder.subList(0, D+1));

                meanVector(subset_mean, dataset, h_prev);
                covarianceMatrix(subset_mean, subset_cov, dataset, h_prev);

                double det = 0;
                //Run C step 3 times. 1 for intiailization from p-set, 2 for the 2 runs after
                for(int i = 0; i < 3; i++)
                    det = MCD_C_step(subset_mean, subset_cov, dataset, h_prev, h, false);
                
                return new Tuple3<>(det, subset_mean, subset_cov);
            }).sorted((o1, o2) -> Double.compare(o1.getX(), o2.getX()))
                .limit(10).collect(Collectors.toList());//get the top 10 best


            
            for(Tuple3<Double, Vec, Matrix> initSolution : top10)
            {
                double prevDev = initSolution.getX();
                
                IntList h_prev = new IntList(h);//This will get populated by the call to C_Step below
                Vec m = initSolution.getY();
                Matrix c = initSolution.getZ();
                for(int iter = 0; iter < 20; iter++)
                {
                    double newDet = MCD_C_step(m, c, dataset, h_prev, h, parallel);
                    if(Math.abs(newDet-prevDev) < 1e-9)//close enough to equal
                        break;
                    prevDev = newDet;
                }
                
                if(prevDev < bestDet)
                {
                    bestCov = c;
                    bestMean = m;
                    bestDet = prevDev;
                }
            }
            
            //return best solution
            
        }
        else//larger set
        {
            int numSplits;//How many sub groups should we produced?
            if(N >= 1500)
                numSplits = 5;
            else
                numSplits = (int) Math.floor(N/300.0);
            //Populate the sub-splits
            IntList randOrderAll = ListUtils.range(0, N);
            Collections.shuffle(randOrderAll, RandomUtil.getLocalRandom());
            IntList[] splits = new IntList[numSplits];
            for(int i = 0; i < numSplits; i++)
                splits[i] = new IntList();
            
            for(int i = 0; i < Math.min(1500, randOrderAll.size()); i++)
                splits[i % splits.length].add(randOrderAll.get(i));
            
            //smaller value of h for each sub set
            int h_sub = (splits[0].size()*h)/N;
            //run process to get top 10 results for each subset 100x times
            
           
            List<Tuple3<Double, Vec, Matrix>> fiftySolutions = 
                    Arrays.asList(splits).stream().flatMap(split ->
            {
                //Create a stream of the top 10 results for each subset
                return ParallelUtils.range(100, parallel)
                .mapToObj(seed ->
                {
                    Random rand = RandomUtil.getRandom(seed);

                    Vec subset_mean = mean.clone();
                    Matrix subset_cov = cov.clone();

                    IntList randOrderSplit = new IntList(split);
                    Collections.shuffle(randOrderSplit, rand);

                    IntList h_prev = new IntList( randOrderSplit.subList(0, D+1));

                    meanVector(subset_mean, dataset, h_prev);
                    covarianceMatrix(subset_mean, subset_cov, dataset, h_prev);

                    double det = 0;
                    //Run C step 3 times. 1 for intiailization from p-set, 2 for the 2 runs after
                    for(int i = 0; i < 3; i++)
                        det = MCD_C_step(subset_mean, subset_cov, dataset, h_prev, h_sub, false);

                    return new Tuple3<>(det, subset_mean, subset_cov);
                }).sorted((o1, o2) -> Double.compare(o1.getX(), o2.getX()))
                    .limit(10);
            }).collect(Collectors.toList());
            
            //"in the merged set, repeat for each of the 50 solutions
            IntSet splits_merged = new IntSet();
            for(int i = 0; i < splits.length; i++)
                splits_merged.addAll(splits[i]);
            int h_merged = (splits_merged.size()*h)/N;
            //do two more steps for each and keep the top 10
            List<Tuple3<Double, Vec, Matrix>> top10 = fiftySolutions.parallelStream().map(tuple->
            {
                Vec subset_mean = tuple.getY();
                Matrix subset_cov = tuple.getZ();
                
                IntList h_prev = new IntList();
                
                double det = 0;
                //Run C step 3 times. 1 for intiailization from p-set, 2 for the 2 runs after
                for(int i = 0; i < 3; i++)
                    det = MCD_C_step(subset_mean, subset_cov, dataset, h_prev, h_merged, false);

                return new Tuple3<>(det, subset_mean, subset_cov);
            }).sorted((o1, o2) -> Double.compare(o1.getX(), o2.getX()))
                    .limit(10)//now we have the top 10 steams
                    .collect(Collectors.toList())
                    ;
            
            for(Tuple3<Double, Vec, Matrix> initSolution : top10)
            {
                double prevDev = initSolution.getX();
                
                IntList h_prev = new IntList(h);//This will get populated by the call to C_Step below
                Vec m = initSolution.getY();
                Matrix c = initSolution.getZ();
                for(int iter = 0; iter < 20; iter++)
                {
                    double newDet = MCD_C_step(m, c, dataset, h_prev, h, parallel);
                    if(Math.abs(newDet-prevDev) < 1e-9)//close enough to equal
                        break;
                    prevDev = newDet;
                }
                
                if(prevDev < bestDet)
                {
                    bestCov = c;
                    bestMean = m;
                    bestDet = prevDev;
                }
            }
        }
        
        
        //Now we have an initial good robust estimate of mean and cov
        
        //To compute correction terms, we need the distances of everyone to the mean
        
        Vec T_full = bestMean;
        Matrix S_full = bestCov;
        
        MahalanobisDistance md = new MahalanobisDistance();
        //regularized cov to ensure its invertable
        LUPDecomposition lup = new LUPDecomposition(S_full.clone());
        //Set inverse matrix for dist
        md.setInverseCovariance(lup.solve(Matrix.eye(S_full.cols())));
        
        ChiSquared chi = new ChiSquared(S_full.cols());
        
        double[] dist = new double[N];
        ParallelUtils.run(parallel, N, (start, end)->
        {
            for(int i = start; i < end; i++)
                dist[i] = md.dist(T_full, dataset.get(i));
        });
        IndexTable it = new IndexTable(dist);
        
        double reScale = Math.pow(dist[it.index(N/2)],2)/chi.invCdf(0.5);
        S_full.mutableMultiply(reScale);
        
        //applyg re-scale to the distsances
        for(int i = 0; i < N; i++)
            dist[i] /= reScale;
        
        //Now we have the corrected Covariance, last step is to detmerine weights and compute mean and cov one last time
        double threshold = Math.sqrt(chi.invCdf(0.975));
        //since weights are 0 or 1, just collect the 1s
        List<Vec> finalSet = new ArrayList<>(N);
        
        for(int i = 0; i < N; i++)
        {
            if(dist[i] <= threshold)
                finalSet.add(dataset.get(i));
        }
        
        //FINAL estimate of mean and cov! 
        mean.zeroOut();
        meanVector(mean, finalSet);
        cov.zeroOut();
        covarianceMatrix(mean, cov, finalSet);
        
        
    }

    /**
     * This helped function implements the C step of the Fast MCD algorithm used
     * by {@link #FastMCD(jsat.linear.Vec, jsat.linear.Matrix, java.util.List, boolean)
     * }.
     *
     * @param subset_mean current estimate of the mean
     * @param subset_cov current estimate of the covariance
     * @param dataset the dataset to work with resept to 
     * @param h_prev a location to store the new subset of used values
     * @param h the subset selection size
     * @param parallel 
     * @return the determinant of the given covariance matrix 
     */
    protected static  <V extends Vec> double MCD_C_step(Vec subset_mean, Matrix subset_cov, List<V> dataset, IntList h_prev, final int h, boolean parallel)
    {
        final int N = dataset.size();
        MahalanobisDistance md = new MahalanobisDistance();
        //regularized cov to ensure its invertable
        for(int i = 0; i < subset_cov.rows(); i++)
            subset_cov.increment(i, i, 1e-4);
        LUPDecomposition lup = new LUPDecomposition(subset_cov.clone());
        //Set inverse matrix for dist
        md.setInverseCovariance(lup.solve(Matrix.eye(subset_cov.cols())));
        double[] dists = new double[N];
        for(int i = 0; i < N; i++)
            dists[i] = md.dist(subset_mean, dataset.get(i));
        //Create new sorted ordering
        IndexTable it = new IndexTable(dists);
        h_prev.clear();
        for(int i = 0; i < h; i++)
            h_prev.add(it.index(i));
        
        //Now lets estimate new mean and cov. We will return the old determinant for lazyness. Worst case is an extra iteration.
        meanVector(subset_mean, dataset, h_prev);
        covarianceMatrix(subset_mean, subset_cov, dataset, h_prev);
        
        return lup.det();
    }
}
