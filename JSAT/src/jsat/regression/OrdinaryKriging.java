package jsat.regression;

import static java.lang.Math.pow;
import java.util.*;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.logging.Level;
import java.util.logging.Logger;
import jsat.classifiers.DataPoint;
import static jsat.linear.DenseVector.toDenseVec;
import jsat.linear.*;
import jsat.parameters.*;
import jsat.utils.SystemInfo;

/**
 * An implementation of Ordinary Kriging with support for a uniform error 
 * measurement. When an {@link #getMeasurementError() error} value is applied, Kriging 
 * becomes equivalent to Gaussian Processes Regression. 
 * 
 * @author Edward Raff
 */
public class OrdinaryKriging implements Regressor, Parameterized
{

	private static final long serialVersionUID = -5774553215322383751L;
	private Variogram vari;
    /**
     * The weight values for each data point
     */
    private Vec X;
    private RegressionDataSet dataSet;
    private double errorSqrd;
    private double nugget;
    
    /**
     * The default nugget value is {@value #DEFAULT_NUGGET}
     */
    public static final double DEFAULT_NUGGET = 0.1;
    /**
     * The default error value is {@link #DEFAULT_ERROR}
     */
    public static final double DEFAULT_ERROR = 0.1;
    
    List<Parameter> params = Collections.unmodifiableList(Parameter.getParamsFromMethods(this));
    
    private Map<String, Parameter> paramMap = Parameter.toParameterMap(params);

    /**
     * Creates a new Ordinary Kriging. 
     * 
     * @param vari the variogram to fit to the data
     * @param error the global measurement error
     * @param nugget the nugget value to add to the variogram
     */
    public OrdinaryKriging(Variogram vari, double error, double nugget)
    {
        this.vari = vari;
        setMeasurementError(error);
        this.nugget = nugget;
    }
    
    /**
     * Creates a new Ordinary Kriging 
     * @param vari the variogram to fit to the data
     * @param error the global measurement error
     */
    public OrdinaryKriging(Variogram vari, double error)
    {
        this(vari, error, DEFAULT_NUGGET);
    }
    
    /**
     * Creates a new Ordinary Kriging with a small error value
     * @param vari the variogram to fit to the data
     */
    public OrdinaryKriging(Variogram vari)
    {
        this(vari, DEFAULT_ERROR);
    }

    /**
     * Creates a new Ordinary Kriging with a small error value using the 
     * {@link PowVariogram power} variogram. 
     */
    public OrdinaryKriging()
    {
        this(new PowVariogram());
    }
    
    @Override
    public double regress(DataPoint data)
    {
        Vec x = data.getNumericalValues();
        int npt = X.length()-1;
        double[] distVals = new double[npt+1];
        for (int i = 0; i < npt; i++)
            distVals[i] = vari.val(x.pNormDist(2, dataSet.getDataPoint(i).getNumericalValues()));
        distVals[npt] = 1.0;
       
        return X.dot(toDenseVec(distVals));
    }

    @Override
    public void train(RegressionDataSet dataSet, ExecutorService threadPool)
    {
        this.dataSet = dataSet;
        /**
         * Size of the data set
         */
        int N = dataSet.getSampleSize();
        /**
         * Stores the target values
         */
        Vec Y = new DenseVector(N+1);
        
        Matrix V = new DenseMatrix(N+1, N+1);
        
        vari.train(dataSet, nugget);
        
        if(threadPool == null)
            setUpVectorMatrix(N, dataSet, V, Y);
        else
            setUpVectorMatrix(N, dataSet, V, Y, threadPool);
        
        for(int i = 0; i < N; i++)
            V.increment(i, i, -errorSqrd);
        
        LUPDecomposition lup;
        if(threadPool == null)
            lup = new LUPDecomposition(V);
        else
            lup = new LUPDecomposition(V, threadPool);
        
        X = lup.solve(Y);
        if(Double.isNaN(lup.det()) || Math.abs(lup.det()) < 1e-5)
        {
            SingularValueDecomposition svd = new SingularValueDecomposition(V);
            X = svd.solve(Y);
        }
    }

    private void setUpVectorMatrix(final int N, RegressionDataSet dataSet, Matrix V, Vec Y)
    {
        for(int i = 0; i < N; i++)
        {
            DataPoint dpi = dataSet.getDataPoint(i);
            Vec xi = dpi.getNumericalValues();
            for(int j = 0; j < N; j++)
            {
                Vec xj = dataSet.getDataPoint(j).getNumericalValues();
                double val = vari.val(xi.pNormDist(2, xj));
                V.set(i, j, val);
                V.set(j, i, val);
            }
            V.set(i, N, 1.0);
            V.set(N, i, 1.0);
            Y.set(i, dataSet.getTargetValue(i));
        }
        V.set(N, N, 0);
    }

    private void setUpVectorMatrix(final int N, final RegressionDataSet dataSet, final Matrix V, final Vec Y, ExecutorService threadPool)
    {
        int pos = 0;
        final CountDownLatch latch = new CountDownLatch(SystemInfo.LogicalCores);
        
        while (pos < SystemInfo.LogicalCores)
        {
            final int id = pos++;
            threadPool.submit(new Runnable() 
            {

                @Override
                public void run()
                {
                    for(int i = id; i < N; i+=SystemInfo.LogicalCores)
                    {
                        DataPoint dpi = dataSet.getDataPoint(i);
                        Vec xi = dpi.getNumericalValues();
                        for(int j = 0; j < N; j++)
                        {
                            Vec xj = dataSet.getDataPoint(j).getNumericalValues();
                            double val = vari.val(xi.pNormDist(2, xj));
                            V.set(i, j, val);
                            V.set(j, i, val);
                        }
                        V.set(i, N, 1.0);
                        V.set(N, i, 1.0);
                        Y.set(i, dataSet.getTargetValue(i));
                    }
                    latch.countDown();
                }
            });
        }
        
        V.set(N, N, 0);
        
        while(pos++ < SystemInfo.LogicalCores)
            latch.countDown();
        
        try
        {
            latch.await();
        }
        catch (InterruptedException ex)
        {
            Logger.getLogger(OrdinaryKriging.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    @Override
    public void train(RegressionDataSet dataSet)
    {
        train(dataSet, null);
    }

    @Override
    public boolean supportsWeightedData()
    {
        return false;
    }

    @Override
    public OrdinaryKriging clone()
    {
        OrdinaryKriging clone = new OrdinaryKriging(vari.clone());
        
        clone.setMeasurementError(getMeasurementError());
        clone.setNugget(getNugget());
        if(this.X != null)
            clone.X = this.X.clone();
        if(this.dataSet != null)
            clone.dataSet = this.dataSet;
        
        return clone;
    }

    /**
     * Sets the measurement error used for Kriging, which is equivalent to 
     * altering the diagonal values of the covariance. While the measurement 
     * errors could be per data point, this implementation provides only a 
     * global error. If the error is set to zero, it will perfectly interpolate
     * all data points. <br>
     * Increasing the error smooths the interpolation, and has a large impact on
     * the regression results. 
     * 
     * @param error the measurement error for all data points
     */
    public void setMeasurementError(double error)
    {
        this.errorSqrd = error*error;
    }

    /**
     * Returns the measurement error used for Kriging, which is equivalent to 
     * altering the diagonal values of the covariance. While the measurement 
     * errors could be per data point, this implementation provides only a 
     * global error. If the error is set to zero, it will perfectly interpolate
     * all data points. 
     * 
     * @return the global error used for the data
     */
    public double getMeasurementError()
    {
        return Math.sqrt(errorSqrd);
    }

    /**
     * Sets the nugget value passed to the variogram during training. The 
     * nugget allows the variogram to start from a non-zero value, and is 
     * equivalent to alerting the off diagonal values of the covariance. <br>
     * Altering the nugget value has only a minor impact on the output
     * 
     * @param nugget the new nugget value
     * @throws ArithmeticException if a negative nugget value is provided
     */
    public void setNugget(double nugget)
    {
        if(nugget < 0 || Double.isNaN(nugget) || Double.isInfinite(nugget))
            throw new ArithmeticException("Nugget must be a positive value");
        this.nugget = nugget;
    }

    /**
     * Returns the nugget value passed to the variogram during training. The 
     * nugget allows the variogram to start from a non-zero value, and is 
     * equivalent to alerting the off diagonal values of the covariance. 
     * 
     * @return the nugget added to the variogram
     */
    public double getNugget()
    {
        return nugget;
    }

    @Override
    public List<Parameter> getParameters()
    {
        return params;
    }

    @Override
    public Parameter getParameter(String paramName)
    {
        return paramMap.get(paramName);
    }
    
    public static interface Variogram extends Cloneable
    {
        /**
         * Sets the values of the variogram 
         * @param dataSet the data set to learn the parameters from
         * @param nugget the nugget value to add tot he variogram, may be 
         * ignored if the variogram want to fit it automatically
         */
        public void train(RegressionDataSet dataSet, double nugget);
        /**
         * Returns the output of the variogram for the given input
         * @param r the input value
         * @return the output of the variogram
         */
        public double val(double r);
        
        public Variogram clone();
    }
    
    public static class PowVariogram implements Variogram
    {
        private double alpha;
        private double beta;

        public PowVariogram()
        {
            this(1.5);
        }

        public PowVariogram(double beta)
        {
            this.beta = beta;
        }

        @Override
        public void train(RegressionDataSet dataSet, double nugget)
        {
            int npt=dataSet.getSampleSize();
            double num=0,denom=0, nugSqrd = nugget*nugget;

            for (int i = 0; i < npt; i++)
            {
                Vec xi = dataSet.getDataPoint(i).getNumericalValues();
                double yi = dataSet.getTargetValue(i);
                for (int j = i + 1; j < npt; j++)
                {
                    Vec xj = dataSet.getDataPoint(j).getNumericalValues();
                    double yj = dataSet.getTargetValue(j);
                    double rb = pow(xi.pNormDist(2, xj), beta);
                    
                    num += rb* (0.5* pow(yi-yj, 2)-nugSqrd);
                    denom += rb*rb;
                }
            }
            alpha = num / denom;
        }
        
        @Override
        public double val(double r)
        {
            return alpha*pow(r, beta);
        }

        @Override
        public Variogram clone()
        {
            PowVariogram clone = new PowVariogram(beta);
            clone.alpha = this.alpha;
            
            return clone;
        }
    }
    
}
