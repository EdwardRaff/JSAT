package jsat.classifiers.linear.kernelized;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import jsat.classifiers.BaseUpdateableClassifier;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.CategoricalResults;
import jsat.classifiers.DataPoint;
import jsat.classifiers.calibration.BinaryScoreClassifier;
import jsat.classifiers.linear.PassiveAggressive;
import jsat.classifiers.neuralnetwork.Perceptron;
import jsat.distributions.kernels.KernelTrick;
import jsat.exceptions.FailedToFitException;
import jsat.linear.DenseMatrix;
import jsat.linear.DenseVector;
import jsat.linear.Matrix;
import jsat.linear.SubMatrix;
import jsat.linear.Vec;
import jsat.parameters.Parameter;
import jsat.parameters.Parameter.ParameterHolder;
import jsat.parameters.Parameterized;
import jsat.utils.DoubleList;

/**
 * An implementation of the Projectron and Projectrion++ algorithms. These are a
 * kernelized extensions of the {@link Perceptron} that bound the number of
 * support vectors used, with the latter incorporating some similarities from
 * {@link PassiveAggressive}. <br>
 * Unlike many other bounded kernel learners, the number of support vectors is
 * not specified by the user. This value is controlled by a sparsity parameter 
 * {@link #setEta(double) }.
 * <br><br>
 * See:
 * <ul>
 * <li>Orabona, F., Keshet, J.,&amp;Caputo, B. (2008). <i>The Projectron: a
 * bounded kernel-based Perceptron</i>. Proceedings of the 25th international
 * conference on Machine learning - ICML ’08 (pp. 720–727). New York, New York,
 * USA: ACM Press. doi:10.1145/1390156.1390247</li>
 * <li>Orabona, F., Keshet, J.,&amp;Caputo, B. (2009). <i>Bounded Kernel-Based
 * Online Learning</i>. The Journal of Machine Learning Research, 10, 2643–2666.
 * </li>
 * </ul>
 *
 * @author Edward Raff
 */
public class Projectron extends BaseUpdateableClassifier implements BinaryScoreClassifier, Parameterized
{

	private static final long serialVersionUID = -4025799790045954359L;
	@ParameterHolder
    private KernelTrick k;
    private double eta;
    /**
     * Marked as "d" in the original papers
     */
    private DoubleList alpha;
    private List<Vec> S;
    private List<Double> cacheAccel;
    private Matrix InvK;
    private Matrix InvKExpanded;
    private double[] k_raw;
    private boolean useMarginUpdates;

    /**
     * Creates a new Projectron++ learner
     *
     * @param k the kernel to use
     */
    public Projectron(KernelTrick k)
    {
        this(k, 0.1);
    }

    /**
     * Creates a new Projectron++ learner
     *
     * @param k the kernel to use
     * @param eta the sparsity parameter
     */
    public Projectron(KernelTrick k, double eta)
    {
        this(k, eta, true);
    }

    /**
     * Creates a new Projectron learner
     *
     * @param k the kernel to use
     * @param eta the sparsity parameter
     * @param useMarginUpdates whether or not to perform projection updates on
     * margin errors
     */
    public Projectron(KernelTrick k, double eta, boolean useMarginUpdates)
    {
        setKernel(k);
        setEta(eta);
        setUseMarginUpdates(useMarginUpdates);
    }

    /**
     * Copy constructor
     *
     * @param toCopy the object to copy
     */
    protected Projectron(Projectron toCopy)
    {
        this.k = toCopy.k.clone();
        this.eta = toCopy.eta;
        if (toCopy.S != null)
        {
            this.alpha = new DoubleList(toCopy.alpha);
            this.S = new ArrayList<Vec>(toCopy.S);
            this.cacheAccel = new DoubleList(toCopy.cacheAccel);
            this.InvKExpanded = toCopy.InvKExpanded.clone();
            this.InvK = new SubMatrix(this.InvKExpanded, 0, 0, toCopy.InvK.rows(), toCopy.InvK.cols());
            this.k_raw = Arrays.copyOf(toCopy.k_raw, toCopy.k_raw.length);
        }
    }

    /**
     * Sets the kernel trick to be used
     *
     * @param k the kernel trick to be use
     */
    public void setKernel(KernelTrick k)
    {
        this.k = k;
    }

    /**
     * Returns the kernel trick in use
     *
     * @return the kernel trick in use
     */
    public KernelTrick getKernel()
    {
        return k;
    }

    /**
     * Sets the &eta; parameter which controls the sparsity of the Projection
     * solution. Larger values result in greater sparsity, at the potential loss
     * of accuracy. If set to 0 and {@link #setUseMarginUpdates(boolean) } is
     * {@code false}, the Projectron degenerates into the standard kernelized
     * Perceptron.
     *
     * @param eta the sparsity parameter in [0, Infinity)
     */
    public void setEta(double eta)
    {
        if (Double.isNaN(eta) || Double.isInfinite(eta) || eta < 0)
            throw new IllegalArgumentException("eta must be in the range [0, Infity), not " + eta);
        this.eta = eta;
    }

    /**
     * Returns the sparsity parameter value
     *
     * @return the sparsity parameter value
     */
    public double getEta()
    {
        return eta;
    }

    /**
     * Sets whether or not projection updates will be performed for margin
     * errors. If {@code true}, this behaves as the Projectrion++ algorithm. If
     * {@code false}, the behavior is equal to the standard Projectron.
     *
     * @param useMarginUpdates {@code true} to perform updates on margin errors
     */
    public void setUseMarginUpdates(boolean useMarginUpdates)
    {
        this.useMarginUpdates = useMarginUpdates;
    }

    /**
     * Returns {@code true} if margin errors can cause updates, {@code false} if
     * not.
     *
     * @return {@code true} if margin errors can cause updates, {@code false} if
     * not.
     */
    public boolean isUseMarginUpdates()
    {
        return useMarginUpdates;
    }

    @Override
    public Projectron clone()
    {
        return new Projectron(this);
    }

    @Override
    public void setUp(CategoricalData[] categoricalAttributes, int numericAttributes, CategoricalData predicting)
    {
        if (numericAttributes < 1)
            throw new IllegalArgumentException("Projectrion requires numeric features");
        else if (predicting.getNumOfCategories() != 2)
            throw new FailedToFitException("Projectrion only supports binary classification");
        final int initSize = 50;
        alpha = new DoubleList(initSize);
        cacheAccel = new DoubleList(initSize);
        S = new ArrayList<Vec>(initSize);
        InvKExpanded = new DenseMatrix(initSize, initSize);
        k_raw = new double[initSize];
    }

    @Override
    public void update(DataPoint dataPoint, int targetClass)
    {
        final Vec x_t = dataPoint.getNumericalValues();
        final List<Double> qi = k.getQueryInfo(x_t);
        final double score = getScore(x_t, qi, k_raw);
        final double y_t = targetClass * 2 - 1;

        //First instance is a special case
        if (S.isEmpty())
        {
            InvK = new SubMatrix(InvKExpanded, 0, 0, 1, 1);
            InvK.set(0, 0, 1.0);
            S.add(x_t);
            alpha.add(y_t);
            cacheAccel.addAll(qi);
            return;
        }
        else if (y_t * score > 1)//No updates needed
            return;
        else if (y_t * score < 1 && y_t * score > 0 && !useMarginUpdates)//margin error but we are ignoring it
            return;

        //Used for both cases, so hoisted out. 
        DenseVector k_t = new DenseVector(k_raw, 0, S.size());
        Vec d = InvK.multiply(k_t);//equation (7)
        final double k_xt = k.eval(0, 0, Arrays.asList(x_t), qi);
        final double k_t_d = k_t.dot(d);
        final double deltaSqrd = Math.max(k_xt - k_t_d, 0);//avoid sqrt(-val)  bellow
        final double delta = Math.sqrt(deltaSqrd);

        if (Math.signum(score) != y_t)
        {
            if (delta < eta)//Project to the basis vectors
            {
                //equation (9)
                for (int i = 0; i < S.size(); i++)
                    alpha.set(i, alpha.get(i) + y_t * d.get(i));
            }
            else//Add to the basis vectors
            {
                //Make sure we have space
                if (S.size() == InvKExpanded.rows())
                {
                    //SubMatrix InvK holds refrence to old one with the correct values
                    InvKExpanded = new DenseMatrix(S.size() * 2, S.size() * 2);
                    for (int i = 0; i < InvK.rows(); i++)
                        for (int j = 0; j < InvK.cols(); j++)
                            InvKExpanded.set(i, j, InvK.get(i, j));
                    InvK = new SubMatrix(InvKExpanded, 0, 0, S.size(), S.size());

                    k_raw = Arrays.copyOf(k_raw, S.size() * 2);
                }
                //Now back to normal
                InvK = new SubMatrix(InvKExpanded, 0, 0, S.size() + 1, S.size() + 1);
                Vec dExp = new DenseVector(S.size() + 1);
                for (int i = 0; i < d.length(); i++)
                    dExp.set(i, d.get(i));
                dExp.set(S.size(), -1);
                if (deltaSqrd > 0)
                    Matrix.OuterProductUpdate(InvK, dExp, dExp, 1 / deltaSqrd);

                S.add(x_t);
                alpha.add(y_t);
                cacheAccel.addAll(qi);
            }

        }
        else if (y_t * score <= 1)//(margin error)
        {
            final double loss = 1 - y_t * score;//y_t*score must be in (0, 1), so no checks needed
            if (loss < delta / eta)
                return;
            //see page 2655
            double tau = Math.max(Math.max(loss / k_t_d, 2 * (loss - delta / eta) / k_t_d), 1.0);

            for (int i = 0; i < S.size(); i++)
                alpha.set(i, alpha.get(i) + y_t * tau * d.get(i));
        }
    }

    @Override
    public CategoricalResults classify(DataPoint data)
    {
        CategoricalResults cr = new CategoricalResults(2);

        if (getScore(data) < 0)
            cr.setProb(0, 1.0);
        else
            cr.setProb(1, 1.0);

        return cr;
    }

    @Override
    public boolean supportsWeightedData()
    {
        return false;
    }

    private double getScore(Vec x, List<Double> qi, final double[] kStore)
    {
        double score = 0;
        for (int i = 0; i < S.size(); i++)
        {
            double tmp = k.eval(i, x, qi, S, cacheAccel);
            if (kStore != null)
                kStore[i] = tmp;
            score += alpha.get(i) * tmp;
        }
        return score;
    }

    @Override
    public double getScore(DataPoint dp)
    {
        return k.evalSum(S, cacheAccel, alpha.getBackingArray(), dp.getNumericalValues(), 0, S.size());
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
