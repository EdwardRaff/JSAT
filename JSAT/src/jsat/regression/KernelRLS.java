package jsat.regression;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutorService;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.DataPoint;
import jsat.distributions.kernels.KernelTrick;
import jsat.linear.*;
import jsat.parameters.Parameter;
import jsat.parameters.Parameter.ParameterHolder;
import jsat.parameters.Parameterized;
import jsat.utils.DoubleList;
import jsat.utils.IntList;
import jsat.utils.ListUtils;

/**
 * Provides an implementation of the Kernel Recursive Least Squares algorithm. 
 * This algorithm updates the model one per data point, and induces sparsity by
 * projecting data points down onto a set of basis vectors learned from the data
 * stream. 
 * <br><br>
 * See: Engel, Y., Mannor, S.,&amp;Meir, R. (2004). <i>The Kernel Recursive 
 * Least-Squares Algorithm</i>. IEEE Transactions on Signal Processing, 52(8), 
 * 2275–2285. doi:10.1109/TSP.2004.830985
 * 
 * @author Edward Raff
 */
public class KernelRLS implements UpdateableRegressor, Parameterized
{

	private static final long serialVersionUID = -7292074388953854317L;
	@ParameterHolder
    private KernelTrick k;
    private double errorTolerance;
    
    private List<Vec> vecs;
    private List<Double> kernelAccel;
    private Matrix K;
    private Matrix InvK;
    private Matrix P;
    
    private Matrix KExpanded;
    private Matrix InvKExpanded;
    private Matrix PExpanded;
    private double[] alphaExpanded;
    
    /**
     * Creates a new Kernel RLS learner
     * @param k the kernel trick to use
     * @param errorTolerance the tolerance for errors in the projection
     */
    public KernelRLS(KernelTrick k, double errorTolerance)
    {
        this.k = k;
        setErrorTolerance(errorTolerance);
    }
    
    /**
     * Copy constructor
     * @param toCopy the object to copy
     */
    protected KernelRLS(KernelRLS toCopy)
    {
        this.k = toCopy.k.clone();
        this.errorTolerance = toCopy.errorTolerance;
        if(toCopy.vecs != null)
        {
            this.vecs = new ArrayList<Vec>(toCopy.vecs.size());
            for(Vec vec : toCopy.vecs)
                this.vecs.add(vec.clone());
        }
        
        if(toCopy.KExpanded != null)
        {
            this.KExpanded = toCopy.KExpanded.clone();
            this.K = new SubMatrix(KExpanded, 0, 0, vecs.size(), vecs.size());
        }
        if(toCopy.InvKExpanded != null)
        {
            this.InvKExpanded = toCopy.InvKExpanded.clone();
            this.InvK = new SubMatrix(InvKExpanded, 0, 0, vecs.size(), vecs.size());
        }
        if(toCopy.PExpanded != null)
        {
            this.PExpanded = toCopy.PExpanded.clone();
            this.P = new SubMatrix(PExpanded, 0, 0, vecs.size(), vecs.size());
        }
        if(toCopy.alphaExpanded != null)
            this.alphaExpanded = Arrays.copyOf(toCopy.alphaExpanded, toCopy.alphaExpanded.length);
    }

    /**
     * Sets the tolerance for errors in approximating a data point by projecting
     * it onto the set of basis vectors. In general: as the tolerance increases 
     * the sparsity of the model increases but the accuracy may go down. 
     * <br>
     * Values in the range 10<sup>x</sup> &forall; x &isin; {-1, -2, -3, -4} 
     * often work well for this algorithm. 
     * 
     * @param v the approximation tolerance
     */
    public void setErrorTolerance(double v)
    {
        if(Double.isNaN(v) || Double.isInfinite(v) || v <= 0)
            throw new IllegalArgumentException("The error tolerance must be a positive constant, not " + v);
        this.errorTolerance = v;
    }

    /**
     * Returns the projection approximation tolerance
     * @return the projection approximation tolerance
     */
    public double getErrorTolerance()
    {
        return errorTolerance;
    }
    
    /**
     * Returns the number of basis vectors that make up the model
     * @return the number of basis vectors that make up the model
     */
    public int getModelSize()
    {
        if(vecs == null)
            return 0;
        return vecs.size();
    }
    
    /**
     * Finalizes the model. During online training, the the gram matrix and its 
     * inverse must be stored to perform updates, at the cost of 
     * O(n<sup>2</sup>) memory. One training is completed, these matrices are no
     * longer needed - and can be removed to reclaim memory by finalizing the 
     * model. Once finalized, the model can no longer be updated - unless reset
     * (destroying the model) by calling 
     * {@link #setUp(jsat.classifiers.CategoricalData[], int) }
     */
    public void finalizeModel()
    {
        alphaExpanded = Arrays.copyOf(alphaExpanded, vecs.size());//dont need extra
        K = KExpanded = InvK = InvKExpanded = P = PExpanded = null;
    }
    
    @Override
    public double regress(DataPoint data)
    {
        final Vec y = data.getNumericalValues();
        
        return k.evalSum(vecs, kernelAccel, alphaExpanded, y, 0, vecs.size());
    }

    @Override
    public void train(RegressionDataSet dataSet, ExecutorService threadPool)
    {
        train(dataSet);
    }

    @Override
    public void train(RegressionDataSet dataSet)
    {
        setUp(dataSet.getCategories(), dataSet.getNumNumericalVars());
        IntList randOrder = new IntList(dataSet.getSampleSize());
        ListUtils.addRange(randOrder, 0, dataSet.getSampleSize(), 1);
        for(int i : randOrder)
            update(dataSet.getDataPoint(i), dataSet.getTargetValue(i));
    }

    @Override
    public boolean supportsWeightedData()
    {
        return false;
    }

    @Override
    public KernelRLS clone()
    {
        return new KernelRLS(this);
    }

    @Override
    public void setUp(CategoricalData[] categoricalAttributes, int numericAttributes)
    {
        vecs = new ArrayList<Vec>();
        if(k.supportsAcceleration())
            kernelAccel = new DoubleList();
        else
            kernelAccel = null;

        K = null;
        InvK = null;
        P = null;

        KExpanded = new DenseMatrix(100, 100);
        InvKExpanded = new DenseMatrix(100, 100);
        PExpanded = new DenseMatrix(100, 100);
        alphaExpanded = new double[100];
    }

    @Override
    public void update(DataPoint dataPoint, final double y_t)
    {
        /*
         * TODO a lot of temporary allocations are done in this code, but 
         * potentially change size - investigate storing them as well. 
         */
        Vec x_t = dataPoint.getNumericalValues();
        
        final List<Double> qi = k.getQueryInfo(x_t);
        final double k_tt = k.eval(0, 0, Arrays.asList(x_t), qi);
        
        if(K == null)//first point to be added
        {
            K = new SubMatrix(KExpanded, 0, 0, 1, 1);
            K.set(0, 0, k_tt);
            InvK = new SubMatrix(InvKExpanded, 0, 0, 1, 1);
            InvK.set(0, 0, 1/k_tt);
            P = new SubMatrix(PExpanded, 0, 0, 1, 1);
            P.set(0, 0, 1);
            alphaExpanded[0] = y_t/k_tt;
            vecs.add(x_t);
            if(kernelAccel != null)
                kernelAccel.addAll(qi);
            return;
        }
        
        
        //Normal case
        DenseVector kxt = new DenseVector(K.rows());

        for (int i = 0; i < kxt.length(); i++)
            kxt.set(i, k.eval(i, x_t, qi, vecs, kernelAccel));

        //ALD test
        final Vec alphas_t = InvK.multiply(kxt);
        final double delta_t = k_tt-alphas_t.dot(kxt);
        final int size = K.rows();
        final double alphaConst = kxt.dot(new DenseVector(alphaExpanded, 0, size));
        if(delta_t > errorTolerance)//add to the dictionary
        {
            vecs.add(x_t);
            if(kernelAccel != null)
                kernelAccel.addAll(qi);
            
            if(size == KExpanded.rows())//we need to grow first
            {
                KExpanded.changeSize(size*2, size*2);
                InvKExpanded.changeSize(size*2, size*2);
                PExpanded.changeSize(size*2, size*2);
                
                alphaExpanded = Arrays.copyOf(alphaExpanded, size*2);
            }
            
            Matrix.OuterProductUpdate(InvK, alphas_t, alphas_t, 1/delta_t);
            K = new SubMatrix(KExpanded, 0, 0, size+1, size+1);
            InvK = new SubMatrix(InvKExpanded, 0, 0, size+1, size+1);
            P = new SubMatrix(PExpanded, 0, 0, size+1, size+1);
            
            //update bottom row and side columns
            for(int i = 0; i < size; i++)
            {
                K.set(size, i, kxt.get(i));
                K.set(i, size, kxt.get(i));
                
                InvK.set(size, i, -alphas_t.get(i)/delta_t);
                InvK.set(i, size, -alphas_t.get(i)/delta_t);
                
                //P is zeros, no change
            }
            //update bottom right corner
            K.set(size, size, k_tt);
            InvK.set(size, size, 1/delta_t);
            P.set(size, size, 1.0);
            
            
            for(int i = 0; i < size; i++)
                alphaExpanded[i] -= alphas_t.get(i)*(y_t-alphaConst)/delta_t;
            alphaExpanded[size] = (y_t-alphaConst)/delta_t;
        }
        else//project onto dictionary
        {
            Vec q_t =P.multiply(alphas_t);
            q_t.mutableDivide(1+alphas_t.dot(q_t));
            
            Matrix.OuterProductUpdate(P, q_t, alphas_t.multiply(P), -1);
            
            Vec InvKqt = InvK.multiply(q_t);
            for(int i = 0; i < size; i++)
                alphaExpanded[i] += InvKqt.get(i)*(y_t-alphaConst);
        }
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
