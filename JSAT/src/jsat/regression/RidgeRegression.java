package jsat.regression;

import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutorService;
import jsat.classifiers.DataPoint;
import jsat.linear.*;
import jsat.parameters.Parameter;
import jsat.parameters.Parameterized;
import jsat.utils.FakeExecutor;

/**
 * An implementation of Ridge Regression that finds the exact solution. Ridge 
 * Regression is equivalent to {@link MultipleLinearRegression} with an added 
 * L<sub>2</sub> penalty for the weight vector. <br><br>
 * Two different methods of finding the solution can be used. This algorithm 
 * should be used only for small dimensions problems with a reasonable number of 
 * example points.<br>
 * For large dimension sparse problems, or dense problems with many data points 
 * (or both), use the {@link StochasticRidgeRegression}. For small data sets 
 * that pose non-linear problems, you can also use {@link KernelRidgeRegression}
 * 
 * @author Edward Raff
 */
public class RidgeRegression implements Regressor, Parameterized
{

	private static final long serialVersionUID = -4605757038780391895L;
	private double lambda;
    private Vec w;
    private double bias;
    private SolverMode mode;
    
    /**
     * Sets which solver to use
     */
    public enum SolverMode
    {
        /**
         * Solves by {@link CholeskyDecomposition}
         */
        EXACT_CHOLESKY,
        /**
         * Solves by {@link SingularValueDecomposition}
         */
        EXACT_SVD,
    }

    public RidgeRegression()
    {
        this(1e-2);
    }
    
    public RidgeRegression(double regularization)
    {
        this(regularization, SolverMode.EXACT_CHOLESKY);
    }
    
    public RidgeRegression(double regularization, SolverMode mode)
    {
        setLambda(regularization);
        setSolverMode(mode);
    }

    /**
     * Sets the regularization parameter used.  
     * @param lambda the positive regularization constant in (0, Inf)
     */
    public void setLambda(double lambda)
    {
        if(Double.isNaN(lambda) || Double.isInfinite(lambda) || lambda <= 0)
            throw new IllegalArgumentException("lambda must be a positive constant, not " + lambda);
        this.lambda = lambda;
    }

    /**
     * Returns the regularization constant in use
     * @return the regularization constant in use 
     */
    public double getLambda()
    {
        return lambda;
    }

    /**
     * Sets which solver is to be used
     * @param mode the solver mode to use 
     */
    public void setSolverMode(SolverMode mode)
    {
        this.mode = mode;
    }

    /**
     * Returns the solver in use
     * @return the solver to use
     */
    public SolverMode getSolverMode()
    {
        return mode;
    }
    
    @Override
    public double regress(DataPoint data)
    {
        Vec x = data.getNumericalValues();
        
        return w.dot(x)+bias;
    }

    @Override
    public void train(RegressionDataSet dataSet, ExecutorService threadPool)
    {
        final int dim = dataSet.getNumNumericalVars()+1;
        DenseMatrix X = new DenseMatrix(dataSet.getSampleSize(), dim);

        for(int i = 0; i < dataSet.getSampleSize(); i++)
        {
            Vec from = dataSet.getDataPoint(i).getNumericalValues();
            X.set(i, 0, 1.0);
            for(int j = 0; j < from.length(); j++)
                X.set(i, j+1, from.get(j));

        }

        final Vec Y = dataSet.getTargetValues();
        final boolean serial = threadPool instanceof FakeExecutor;

        if(mode == SolverMode.EXACT_SVD)
        {
            SingularValueDecomposition svd = new SingularValueDecomposition(X);
            double[] ridgeD;
            ridgeD = Arrays.copyOf(svd.getSingularValues(), dim);
            for(int i = 0; i < ridgeD.length; i++)
                ridgeD[i] = 1 / (Math.pow(ridgeD[i], 2)+lambda);
            Matrix U = svd.getU();
            Matrix V = svd.getV();


            // w = V (D^2 + lambda I)^(-1) D U^T y
            Matrix.diagMult(V, DenseVector.toDenseVec(ridgeD));
            Matrix.diagMult(V, DenseVector.toDenseVec(svd.getSingularValues()));
            w = V.multiply(U.transpose()).multiply(Y);
        }
        else//cholesky
        {
            
            Matrix H = serial ? X.transposeMultiply(X) : X.transposeMultiply(X, threadPool);
            //H + I * reg     equiv to H.mutableAdd(Matrix.eye(H.rows()).multiply(regularization));
            for(int i = 0; i < H.rows(); i++)
                H.increment(i, i, lambda);
            CholeskyDecomposition cd = serial ? new CholeskyDecomposition(H) : new CholeskyDecomposition(H, threadPool);
            w = cd.solve(Matrix.eye(H.rows())).multiply(X.transpose()).multiply(Y);
        }
        
        //reformat w and seperate out bias term
        bias = w.get(0);
        Vec newW = new DenseVector(w.length()-1);
        for(int i = 0; i < newW.length(); i++)
            newW.set(i, w.get(i+1));
        w = newW;
    }

    @Override
    public void train(RegressionDataSet dataSet)
    {
        train(dataSet, new FakeExecutor());
    }

    @Override
    public boolean supportsWeightedData()
    {
        return false;
    }

    @Override
    public RidgeRegression clone()
    {
        RidgeRegression clone = new RidgeRegression(lambda);
        if(this.w != null)
            clone.w = this.w.clone();
        clone.bias = this.bias;
        return clone;
    }

    @Override
    public List<Parameter> getParameters()
    {
        return Parameter.getParamsFromMethods(this);
    }

    @Override
    public Parameter getParameter(String paramName)
    {
        return Parameter.toParameterMap(getParameters()).get(paramName);
    }
    
}
