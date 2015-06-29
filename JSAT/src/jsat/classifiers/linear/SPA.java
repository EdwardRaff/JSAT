package jsat.classifiers.linear;

import static java.lang.Math.*;
import java.util.*;
import jsat.DataSet;
import jsat.SimpleWeightVectorModel;
import jsat.classifiers.*;
import jsat.distributions.Distribution;
import jsat.linear.DenseVector;
import jsat.linear.Vec;
import jsat.parameters.Parameter;
import jsat.parameters.Parameterized;
import jsat.utils.IndexTable;

/**
 * Support class Passive Aggressive (SPA) is a multi class generalization of 
 * {@link PassiveAggressive}. It works in the same philosophy, and can obtain 
 * better multi class accuracy then PA  used with a meta learner. <br>
 * SPA is more sensitive to small values for the {@link #setC(double) 
 * aggressiveness parameter}. <br>
 * If working with a binary classification problem, SPA reduces to PA, and the
 * original PA implementation should be used instead. <br>
 * By default, the {@link #setUseBias(boolean) biast term} is not used. 
 * <br><br>
 * See: <br>
 * Matsushima, S., Shimizu, N., Yoshida, K., Ninomiya, T.,&amp;Nakagawa, H. 
 * (2010). <i>Exact Passive-Aggressive Algorithm for Multiclass Classification 
 * Using Support Class</i>. SIAM International Conference on Data Mining - SDM
 * (pp. 303–314). Retrieved from 
 * <a href="https://www.siam.org/proceedings/datamining/2010/dm10_027_matsushimas.pdf">here</a>
 * 
 * @author Edward Raff
 */
public class SPA extends BaseUpdateableClassifier implements Parameterized, SimpleWeightVectorModel
{

    private static final long serialVersionUID = 3613279663279244169L;
    private Vec[] w;
    private double[] bias;
    private double C = 1;
    private boolean useBias = false;
    private PassiveAggressive.Mode mode;

    /**
     * Creates a new Passive Aggressive learner that does 10 epochs and uses
     * PA2. 
     */
    public SPA()
    {
        this(10, PassiveAggressive.Mode.PA2);
    }
    
    /**
     * Creates a new Passive Aggressive learner
     * 
     * @param epochs the number of training epochs to use during batch training
     * @param mode which version of the update to perform 
     */
    public SPA(int epochs, PassiveAggressive.Mode mode)
    {
        setEpochs(epochs);
        setMode(mode);
    }

    /**
     * Sets whether or not the implementation will use an implicit bias term 
     * appended to the inputs or not. 
     * @param useBias {@code true} to add an implicit bias term, {@code false} 
     * to use the data as given
     */
    public void setUseBias(boolean useBias)
    {
        this.useBias = useBias;
    }

    /**
     * Returns true if an implicit bias term will be added, false otherwise
     * @return true if an implicit bias term will be added, false otherwise
     */
    public boolean isUseBias()
    {
        return useBias;
    }
    
    /**
     * Set the aggressiveness parameter. Increasing the value of this parameter 
     * increases the aggressiveness of the algorithm. It must be a positive 
     * value. This parameter essentially performs a type of regularization on 
     * the updates
     * <br>
     * An infinitely large value is equivalent to being completely aggressive, 
     * and is performed when the mode is set to {@link PassiveAggressive.Mode#PA}. 
     * 
     * @param C the positive aggressiveness parameter
     */
    public void setC(double C)
    {
        if(Double.isNaN(C) || Double.isInfinite(C) || C <= 0)
            throw new ArithmeticException("Aggressiveness must be a positive constant");
        this.C = C;
    }
    
    /**
     * Returns the aggressiveness parameter 
     * @return the aggressiveness parameter 
     */
    public double getC()
    {
        return C;
    }

    /**
     * Sets which version of the PA update is used. 
     * @param mode which PA update style to perform
     */
    public void setMode(PassiveAggressive.Mode mode)
    {
        this.mode = mode;
    }

    /**
     * Returns which version of the PA update is used
     * @return which PA update style is used
     */
    public PassiveAggressive.Mode getMode()
    {
        return mode;
    }

    @Override
    public Vec getRawWeight(int index)
    {
        return w[index];
    }

    @Override
    public double getBias(int index)
    {
        return bias[index];
    }
    
    @Override
    public int numWeightsVecs()
    {
        return w.length;
    }
    
    @Override
    public SPA clone()
    {
        SPA clone = new SPA();
        if(this.w != null)
        {
            clone.w = new Vec[this.w.length];
            for(int i = 0; i < w.length; i++)
                clone.w[i] = this.w[i].clone();
        }
        if(this.it != null)
            clone.it = new IndexTable(this.it.length());
        if(this.loss != null)
            clone.loss = Arrays.copyOf(this.loss, this.loss.length);
        clone.C = this.C;
        clone.mode = this.mode;
        if(this.bias != null)
            clone.bias = Arrays.copyOf(this.bias, this.bias.length);
        clone.useBias = this.useBias;
        return clone;
    }

    @Override
    public void setUp(CategoricalData[] categoricalAttributes, int numericAttributes, CategoricalData predicting)
    {
        w = new Vec[predicting.getNumOfCategories()];
        for(int i = 0; i < w.length; i++)
            w[i] = new DenseVector(numericAttributes);
        bias = new double[w.length];
        loss = new double[w.length];
        it = new IndexTable(w.length);
    }

    private double[] loss;
    private IndexTable it;
    
    /**
     * Part A of SPA algorithm
     * @param xNorm the value of the squared 2 norm training input
     * @param k the value of k
     * @param loss_k the loss of the k'th sorted value
     * @return the target support class goal to be less than
     */
    private double getSupportClassGoal(final double xNorm, final int k, final double loss_k)
    {
        if(mode == PassiveAggressive.Mode.PA1)
            return min((k-1)*loss_k+C*xNorm, k*loss_k);
        else if(mode == PassiveAggressive.Mode.PA2)
            return ((k*xNorm+(k-1)/(2*C))/(xNorm+1.0/(2*C)))*loss_k;
        else
            return k*loss_k;
    }
    
    /**
     * Part B of SPA algorithm
     * @param loss_cur the loss for the current value in consideration
     * @param xNorm the value of the squared 2 norm training input
     * @param k the value of k (number of support classes +1)
     * @param supLossSum the sum of the loss for the support classes
     * @return the update step size
     */
    private double getStepSize(final double loss_cur, final double xNorm, int k, final double supLossSum)
    {
        if(mode == PassiveAggressive.Mode.PA1)
            return max(0, loss_cur-max(supLossSum/(k-1)-C/(k-1)*xNorm, supLossSum/k))/xNorm;
        else if(mode == PassiveAggressive.Mode.PA2)
            return max(0, loss_cur-(xNorm+1/(2*C))/(k*xNorm+(k-1)/(2*C))*supLossSum )/xNorm;
        else
            return max(0, loss_cur-supLossSum/k)/xNorm;
    }
    
    @Override
    public void update(DataPoint dataPoint, int targetClass)
    {
        Vec x = dataPoint.getNumericalValues();
        final double w_y_dot_x = w[targetClass].dot(x) + bias[targetClass];
        for (int v = 0; v < w.length; v++)
            if (v != targetClass)
                loss[v] = max(0, 1 - (w_y_dot_x - w[v].dot(x) - bias[v]));
            else
                loss[v] = Double.POSITIVE_INFINITY;//set in Inft so its ends up in index 0, and gets skipped
        final double xNorm = pow(x.pNorm(2) + (useBias ? 1 : 0), 2);

        it.sortR(loss);

        int k = 1;

        double T31 = 0;//Theorem 3.1 

        while (k < loss.length && T31 < getSupportClassGoal(xNorm, k, loss[it.index(k)]))
            T31 += loss[it.index(k++)];

        double supportLossSum = 0;
        for (int j = 1; j < k; j++)
            supportLossSum += loss[it.index(j)];

        for (int j = 1; j < k; j++)
        {
            final int v = it.index(j);
            double tau = getStepSize(loss[v], xNorm, k, supportLossSum);
            w[targetClass].mutableAdd(tau, x);
            w[v].mutableSubtract(tau, x);
            if (useBias)
            {
                bias[targetClass] += tau;
                bias[v] -= tau;
            }
        }
    }

    @Override
    public CategoricalResults classify(DataPoint data)
    {
        Vec x = data.getNumericalValues();
        CategoricalResults cr = new CategoricalResults(w.length);
        int maxIdx = 0;
        double maxVAl = w[0].dot(x)+bias[0];
        for(int i = 1; i < w.length; i++)
        {
            double val = w[i].dot(x)+bias[i];
            if(val > maxVAl)
            {
                maxVAl = val;
                maxIdx = i;
            }
        }
        cr.setProb(maxIdx, 1.0);
        return cr;
    }

    @Override
    public boolean supportsWeightedData()
    {
        return false;
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
    
    /**
     * Guess the distribution to use for the regularization term
     * {@link #setC(double) C} in Support PassiveAggressive.
     *
     * @param d the data set to get the guess for
     * @return the guess for the C parameter 
     */
    public static Distribution guessC(DataSet d)
    {
        return PassiveAggressive.guessC(d);
    }
}
