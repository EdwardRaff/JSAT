
package jsat.regression;

import jsat.SingleWeightVectorModel;
import jsat.classifiers.DataPoint;
import jsat.classifiers.DataPointPair;
import jsat.linear.DenseMatrix;
import jsat.linear.DenseVector;
import jsat.linear.Matrix;
import jsat.linear.QRDecomposition;
import jsat.linear.Vec;
import jsat.utils.concurrent.ParallelUtils;

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

    public MultipleLinearRegression(boolean useWeights)
    {
        this.useWeights = useWeights;
    }
    
    @Override
    public double regress(DataPoint data)
    {
        return B.dot(data.getNumericalValues())+a;
    }

    @Override
    public void train(RegressionDataSet dataSet, boolean parallel)
    {
        if(dataSet.getNumCategoricalVars() > 0)
            throw new RuntimeException("Multiple Linear Regression only works with numerical values");
        int sda = dataSet.size();
        DenseMatrix X = new DenseMatrix(dataSet.size(), dataSet.getNumNumericalVars()+1);
        DenseVector Y = new DenseVector(dataSet.size());
        
        
        //Construct matrix and vector, Y = X * B, we will solve for B or its least squares solution
        for(int i = 0; i < dataSet.size(); i++)
        {
            DataPointPair<Double> dpp = dataSet.getDataPointPair(i);
            
            Y.set(i, dpp.getPair());
            X.set(i, 0, 1.0);//First column is all ones
            Vec vals = dpp.getVector();
            for(int j = 0; j < vals.length(); j++)
                X.set(i, j+1, vals.get(j));
        }
        
        if(useWeights)
        {
            //The sqrt(weight) vector can be applied to X and Y, and then QR can procede as normal 
            Vec weights = new DenseVector(dataSet.size());
            for(int i = 0; i < dataSet.size(); i++)
                weights.set(i, Math.sqrt(dataSet.getWeight(i)));
            
            Matrix.diagMult(weights, X);
            Y.mutablePairwiseMultiply(weights);
        }
        
        Matrix[] QR = parallel ? X.qr(ParallelUtils.CACHED_THREAD_POOL) : X.qr();
        
        QRDecomposition qrDecomp = new QRDecomposition(QR[0], QR[1]);
        
        Vec tmp = qrDecomp.solve(Y);
        a = tmp.get(0);
        B = new DenseVector(dataSet.getNumNumericalVars());
        for(int i = 1; i < tmp.length(); i++)
            B.set(i-1, tmp.get(i));
        
    }

    @Override
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
    public Vec getRawWeight(int index)
    {
        if(index < 1)
            return getRawWeight();
        else
            throw new IndexOutOfBoundsException("Model has only 1 weight vector");
    }

    @Override
    public double getBias(int index)
    {
        if (index < 1)
            return getBias();
        else
            throw new IndexOutOfBoundsException("Model has only 1 weight vector");
    }
    
    @Override
    public int numWeightsVecs()
    {
        return 1;
    }
    
    @Override
    public MultipleLinearRegression clone()
    {
        MultipleLinearRegression copy = new MultipleLinearRegression();
        if(B != null)
            copy.B = this.B.clone();
        copy.a = this.a;
        
        return copy;
    }
    
}
