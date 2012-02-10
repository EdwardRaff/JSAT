
package jsat.distributions.multivariate;

import java.util.List;
import jsat.classifiers.DataPoint;
import jsat.linear.DenseMatrix;
import jsat.linear.DenseVector;
import jsat.linear.LUPDecomposition;
import jsat.linear.Matrix;
import jsat.linear.MatrixStatistics;
import jsat.linear.Vec;
import static java.lang.Math.*;
import static jsat.linear.MatrixStatistics.*;

/**
 * Class for the multivariate Normal distribution. It is often called the Multivariate Gaussian distribution. 
 * 
 * @author Edward Raff
 */
public class NormalM extends MultivariateDistributionSkeleton
{
    /**
     * When computing the PDF of some x, part of the equation is only dependent on the covariance matrix. This part is
     * <pre>
     *       -k
     *       --          -1
     *        2          --
     * /  __\             2
     * \2 ||/   (|Sigma|)
     * </pre>
     * where k is the dimension, Sigma is the covariance matrix, and || denotes the determinant. <br>
     * Taking the log of this gives
     * <pre>
     *         /  __\
     * (-k) log\2 ||/ - log(|Sigma|)
     * -----------------------------
     *               2
     * </pre>
     */
    private double logPDFConst;
    /**
     * When we compute the constant {@link #logPDFConst}, we only need the inverse of the covariance matrix. 
     */
    private Matrix invCovariance;
    private Vec mean;

    public NormalM(Vec mean, Matrix covariance)
    {
        setMeanCovariance(mean, covariance);
    }

    public NormalM()
    {
    }
    
    /**
     * Sets the mean and covariance for this distribution. For an <i>n</i> dimensional distribution,
     * <tt>mean</tt> should be of length <i>n</i> and <tt>covariance</tt> should be an <i>n</i> by <i>n</i> matrix.
     * It is also a requirement that the matrix be symmetric positive definite. 
     * @param mean the mean for the distribution. A copy will be used. 
     * @param covariance the covariance for this distribution. A copy will be used. 
     * @throws ArithmeticException if the <tt>mean</tt> and <tt>covariance</tt> do not agree, or the covariance is not 
     * positive definite. An exception may not be throw for all bad matrices. 
     */
    public void setMeanCovariance(Vec mean, Matrix covariance)
    {
        if(!covariance.isSquare())
            throw new ArithmeticException("Covariance matrix must be square");
        else if(mean.length() != covariance.rows())
            throw new ArithmeticException("The mean vector and matrix must have the same dimension," +
                    mean.length() + " does not match [" + covariance.rows() + ", " + covariance.rows() +"]" );
        //Else, we are good!
        this.mean = mean.clone();
        setCovariance(covariance);
    }
    
    /**
     * Sets the covariance matrix for this matrix. 
     * @param covMatrix set the covariance matrix used for this distribution
     * @throws ArithmeticException if the covariance matrix is not square, 
     * does not agree with the mean, or is not positive definite.  An 
     * exception may not be throw for all bad matrices. 
     */
    public void setCovariance(Matrix covMatrix)
    {
        if(!covMatrix.isSquare())
            throw new ArithmeticException("Covariance matrix must be square");
        else if(covMatrix.rows() != this.mean.length())
            throw new ArithmeticException("Covariance matrix does not agree with the mean");
        
        LUPDecomposition lup = new LUPDecomposition(covMatrix.clone());
        int k = mean.length();
        double det = lup.det();
        if(det <= 0)//Ehhh... should zero be handled differently? I'm not sure
            throw new ArithmeticException("An invalid covariance matrix was given. Covariance matrix must be symmetric positive definite");
        this.logPDFConst = (-k*log(2*PI)-log(det))*0.5;
        this.invCovariance = lup.solve(Matrix.eye(k));
    }

    @Override
    public double logPdf(Vec x)
    {
        if(mean == null)
            throw new ArithmeticException("No mean or variance set");
        Vec xMinusMean = x.subtract(mean);
        //Compute the part that is depdentent on x
        double xDependent = xMinusMean.dot(invCovariance.multiply(xMinusMean))*-0.5;
        return logPDFConst + xDependent;
    }

    @Override
    public double pdf(Vec x)
    {
        return exp(logPdf(x));
    }

    @Override
    public boolean setUsingData(List<Vec> dataSet)
    {
        Vec origMean = this.mean;
        try
        {
            Vec newMean = MatrixStatistics.MeanVector(dataSet);
            Matrix covariance = MatrixStatistics.CovarianceMatrix(mean, dataSet);

            this.mean = newMean;
            setCovariance(covariance);
            return true;
        }
        catch(ArithmeticException ex)
        {
            this.mean = origMean;
            return false;
        }
    }

    @Override
    public boolean setUsingDataList(List<DataPoint> dataSet)
    {
        Vec origMean = this.mean;
        try
        {
            Vec newMean = new DenseVector(dataSet.get(0).getNumericalValues().length());
            double sumOfWeights = 0.0, sumOfSquaredWeights = 0.0;
            for(int i = 0; i < dataSet.size(); i++)
            {
                DataPoint dp = dataSet.get(i);
                newMean.mutableAdd(dp.getWeight(), dp.getNumericalValues());
                sumOfWeights += dp.getWeight();
                sumOfSquaredWeights += Math.pow(dp.getWeight(), 2);
            }
            newMean.mutableDivide(sumOfWeights);

            //Now compute the covariance matrix
            Matrix covariance = new DenseMatrix(newMean.length(), newMean.length());
            CovarianceMatrix(newMean, dataSet, covariance, sumOfWeights, sumOfSquaredWeights);

            this.mean = newMean;
            setCovariance(covariance);
            return true;
        }
        catch(ArithmeticException ex)
        {
            this.mean = origMean;
            return false;
        }
    }

    @Override
    public NormalM clone()
    {
        NormalM clone = new NormalM();
        if(this.invCovariance != null)
            clone.invCovariance = this.invCovariance.clone();
        if(this.mean != null)
            clone.mean = this.mean.clone();
        clone.logPDFConst = this.logPDFConst;
        return clone;
    }
}
