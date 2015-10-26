
package jsat.regression;

import java.util.concurrent.ExecutorService;
import jsat.SingleWeightVectorModel;
import jsat.classifiers.DataPoint;
import jsat.classifiers.DataPointPair;
import jsat.linear.DenseMatrix;
import jsat.linear.DenseVector;
import jsat.linear.Matrix;
import jsat.linear.QRDecomposition;
import jsat.linear.Vec;
import jsat.utils.FakeExecutor;

/**
 *
 * @author Edward Raff
 */
public class MultipleLinearRegression implements Regressor, SingleWeightVectorModel
{

	private static final long serialVersionUID = 7694194181910565061L;
	/**
     * The vector B such that Y = X * B is the least squares solution. Will be stored as Y = X * B + a
     */
    private Vec B;
    /**
     * The offset value that is not multiplied by any variable 
     */
    private double a;
    private boolean useWeights = false;

    public MultipleLinearRegression()
    {
        this(true);
    }

    public MultipleLinearRegression(final boolean useWeights)
    {
        this.useWeights = useWeights;
    }
    
    public double regress(final DataPoint data)
    {
        return B.dot(data.getNumericalValues())+a;
    }

    public void train(final RegressionDataSet dataSet, final ExecutorService threadPool)
    {
        if(dataSet.getNumCategoricalVars() > 0) {
          throw new RuntimeException("Multiple Linear Regression only works with numerical values");
        }
        final int sda = dataSet.getSampleSize();
        final DenseMatrix X = new DenseMatrix(dataSet.getSampleSize(), dataSet.getNumNumericalVars()+1);
        final DenseVector Y = new DenseVector(dataSet.getSampleSize());
        
        
        //Construct matrix and vector, Y = X * B, we will solve for B or its least squares solution
        for(int i = 0; i < dataSet.getSampleSize(); i++)
        {
            final DataPointPair<Double> dpp = dataSet.getDataPointPair(i);
            
            Y.set(i, dpp.getPair());
            X.set(i, 0, 1.0);//First column is all ones
            final Vec vals = dpp.getVector();
            for(int j = 0; j < vals.length(); j++) {
              X.set(i, j+1, vals.get(j));
            }
        }
        
        if(useWeights)
        {
            //The sqrt(weight) vector can be applied to X and Y, and then QR can procede as normal 
            final Vec weights = new DenseVector(dataSet.getSampleSize());
            for(int i = 0; i < dataSet.getSampleSize(); i++) {
              weights.set(i, Math.sqrt(dataSet.getDataPoint(i).getWeight()));
            }
            
            Matrix.diagMult(weights, X);
            Y.mutablePairwiseMultiply(weights);
        }
        
        final Matrix[] QR = X.qr(threadPool);
        
        final QRDecomposition qrDecomp = new QRDecomposition(QR[0], QR[1]);
        
        final Vec tmp = qrDecomp.solve(Y);
        a = tmp.get(0);
        B = new DenseVector(dataSet.getNumNumericalVars());
        for(int i = 1; i < tmp.length(); i++) {
          B.set(i-1, tmp.get(i));
        }
        
    }

    public void train(final RegressionDataSet dataSet)
    {
        train(dataSet, new FakeExecutor());
    }

    public boolean supportsWeightedData()
    {
        return useWeights;
    }

    @Override
    public Vec getRawWeight()
    {
        return B;
    }

    @Override
    public double getBias()
    {
        return a;
    }

    @Override
    public Vec getRawWeight(final int index)
    {
        if(index < 1) {
          return getRawWeight();
        } else {
          throw new IndexOutOfBoundsException("Model has only 1 weight vector");
        }
    }

    @Override
    public double getBias(final int index)
    {
        if (index < 1) {
          return getBias();
        } else {
          throw new IndexOutOfBoundsException("Model has only 1 weight vector");
        }
    }
    
    @Override
    public int numWeightsVecs()
    {
        return 1;
    }
    
    @Override
    public MultipleLinearRegression clone()
    {
        final MultipleLinearRegression copy = new MultipleLinearRegression();
        if(B != null) {
          copy.B = this.B.clone();
        }
        copy.a = this.a;
        
        return copy;
    }
    
}
