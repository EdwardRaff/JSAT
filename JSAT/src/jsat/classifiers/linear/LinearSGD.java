package jsat.classifiers.linear;

import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutorService;
import jsat.SimpleWeightVectorModel;
import jsat.classifiers.BaseUpdateableClassifier;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.CategoricalResults;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.DataPoint;
import jsat.exceptions.FailedToFitException;
import jsat.linear.DenseVector;
import jsat.linear.IndexValue;
import jsat.linear.ScaledVector;
import jsat.linear.Vec;
import jsat.lossfunctions.HingeLoss;
import jsat.lossfunctions.LossC;
import jsat.lossfunctions.LossFunc;
import jsat.lossfunctions.LossMC;
import jsat.lossfunctions.LossR;
import jsat.math.decayrates.DecayRate;
import jsat.math.decayrates.PowerDecay;
import jsat.parameters.Parameter;
import jsat.parameters.Parameterized;
import jsat.regression.BaseUpdateableRegressor;
import jsat.regression.RegressionDataSet;
import jsat.regression.UpdateableRegressor;

/**
 * LinearSGD learns either a classification or regression problem depending on 
 * the {@link #setLoss(jsat.lossfunctions.LossFunc) loss function &#x2113;(w,x)}
 * used. The solution attempts to minimize 
 * <big>&sum;</big><sub>i</sub> &#x2113;(w,x<sub>i</sub>) + 
 * {@link #setLambda0(double) &lambda<sub>0</sub>}/2 ||w||<sub>2</sub><sup>2</sup> + 
 * {@link #setLambda1(double) &lambda<sub>1</sub>} ||w||<sub>1</sub>, and is 
 * trained by Stochastic Gradient Descent. <br>
 * <br>
 * <br>
 * NOTE: To support L<sub>1</sub> regularization with sparse results and online 
 * learning at the same time, the normalization of the regularization penalty by
 * the number of data points is not present in the implementation at this time. 
 * Setting {@link #setLambda1(double) &lambda<sub>1</sub>} to the desired value 
 * divided by the number of unique data points in the whole set will result in 
 * the correct regularization penalty being applied. 
 * 
 * See:
 * <ul>
 * <li>Tsuruoka, Y., Tsujii, J., & Ananiadou, S. (2009). <i>Stochastic gradient 
 * descent training for L1-regularized log-linear models with cumulative 
 * penalty</i>. Proceedings of the Joint Conference of the 47th Annual Meeting 
 * of the ACL and the 4th International Joint Conference on Natural Language 
 * Processing of the AFNLP, 1, 477. doi:10.3115/1687878.1687946</li>
 * </ul>
 * @author Edward Raff
 */
public class LinearSGD extends BaseUpdateableClassifier implements UpdateableRegressor, Parameterized, SimpleWeightVectorModel
{
    
    private LossFunc loss;
    private double eta;
    private DecayRate decay;
    private Vec[] ws;
    private double[] bs;
    private int time;
    private double lambda0;
    private double lambda1;
    private double l1U;
    private double[][] l1Q;
    private boolean useBias = true;

    /**
     * Creates a new LinearSGD learner for multi-class classification problems. 
     */
    public LinearSGD()
    {
        this(new HingeLoss(), 1e-4, 0.0);
    }

    /**
     * Creates a new LinearSGD learner
     * @param loss the loss function to use
     * @param lambda0 the L<sub>2</sub> regularization term
     * @param lambda1 the L<sub>1</sub> regularization term
     */
    public LinearSGD(LossFunc loss, double lambda0, double lambda1)
    {
        this(loss, 0.001, new PowerDecay(1, 0.1), lambda0, lambda1);
    }

    /**
     * Creates a new LinearSGD learner. 
     * @param loss the loss function to use
     * @param eta the initial learning rate
     * @param decay the decay rate for &eta;
     * @param lambda0 the L<sub>2</sub> regularization term
     * @param lambda1 the L<sub>1</sub> regularization term
     */
    public LinearSGD(LossFunc loss, double eta, DecayRate decay, double lambda0, double lambda1)
    {
        setLoss(loss);
        setEta(eta);
        setEtaDecay(decay);
        setLambda0(lambda0);
        setLambda1(lambda1);
    }


    /**
     * Copy constructor
     * @param toClone the object to copy
     */
    public LinearSGD(LinearSGD toClone)
    {
        this.loss = toClone.loss.clone();
        this.eta = toClone.eta;
        this.decay = toClone.decay.clone();
        this.time = toClone.time;
        this.lambda0 = toClone.lambda0;
        this.lambda1 = toClone.lambda1;
        this.l1U = toClone.l1U;
        if(toClone.l1Q != null)
        {
            this.l1Q = new double[toClone.l1Q.length][];
            for(int i = 0; i < toClone.l1Q.length; i++)
                this.l1Q[i] = Arrays.copyOf(toClone.l1Q[i], toClone.l1Q[i].length);
        }
        if(toClone.ws != null)
        {
            this.ws = new Vec[toClone.ws.length];
            this.bs = new double[toClone.bs.length];
            for(int i = 0; i < ws.length; i++)
            {
                this.ws[i] = toClone.ws[i].clone();
                this.bs[i] = toClone.bs[i];
            }
        }
    }

    /**
     * Sets the rate at which {@link #setEta(double) &eta;} is decayed at each 
     * update. 
     * @param decay the decay rate to use
     */
    public void setEtaDecay(DecayRate decay)
    {
        this.decay = decay;
    }

    /**
     * Returns the decay rate in use
     * @return the decay rate in use
     */
    public DecayRate getEtaDecay()
    {
        return decay;
    }

    /**
     * Sets the initial learning rate &eta; to use. It should generally be in 
     * (0, 1), but any positive value is acceptable. 
     * @param eta the learning rate to use. 
     */
    public void setEta(double eta)
    {
        if(eta <= 0 || Double.isNaN(eta) || Double.isInfinite(eta))
            throw new IllegalArgumentException("eta must be a positive constant, not " + eta);
        this.eta = eta;
    }

    /**
     * Returns the current learning rate in use
     * @return the current learning rate in use
     */
    public double getEta()
    {
        return eta;
    }

    /**
     * Sets the loss function used for the model. The loss function controls 
     * whether or not regression, binary classification, or multi-class 
     * classification is supported. 
     * @param loss the loss function to use
     */
    public void setLoss(LossFunc loss)
    {
        this.loss = loss;
    }

    /**
     * Returns the loss function in use
     * @return the loss function in use
     */
    public LossFunc getLoss()
    {
        return loss;
    }

    /**
     * &lambda;<sub>0</sub> controls the L<sub>2</sub> regularization penalty. 
     * @param lambda0 the L<sub>2</sub> regularization penalty to use
     */
    public void setLambda0(double lambda0)
    {
        if(lambda0 < 0 || Double.isNaN(lambda0) || Double.isInfinite(lambda0))
            throw new IllegalArgumentException("Lambda0 must be non-negative, not " + lambda0);
        this.lambda0 = lambda0;
    }

    /**
     * Returns the L<sub>2</sub> regularization term in use
     * @return the L<sub>2</sub> regularization term in use
     */
    public double getLambda0()
    {
        return lambda0;
    }

    /**
     * &lambda;<sub>1</sub> controls the L<sub>1</sub> regularization penalty. 
     * @param lambda1 the L<sub>1</sub> regularization penalty to use
     */
    public void setLambda1(double lambda1)
    {
        if(lambda1 < 0 || Double.isNaN(lambda1) || Double.isInfinite(lambda1))
            throw new IllegalArgumentException("Lambda1 must be non-negative, not " + lambda1);
        this.lambda1 = lambda1;
    }

    /**
     * Returns the L<sub>1</sub> regularization term in use
     * @return the L<sub>1</sub> regularization term in use
     */
    public double getLambda1()
    {
        return lambda1;
    }
    
    /**
     * Sets whether or not an implicit bias term will be added to the data set
     * @param useBias {@code true} to add an implicit bias term
     */
    public void setUseBias(boolean useBias)
    {
        this.useBias = useBias;
    }

    /**
     * Returns whether or not an implicit bias term is in use
     * @return {@code true} if a bias term is in use
     */
    public boolean isUseBias()
    {
        return useBias;
    }
    
    @Override
    public LinearSGD clone()
    {
        return new LinearSGD(this);
    }

    @Override
    public void setUp(CategoricalData[] categoricalAttributes, int numericAttributes, CategoricalData predicting)
    {
        if(!(loss instanceof LossC))
            throw new FailedToFitException("Loss function " + loss.getClass().getSimpleName() + " only supports regression");
        if(predicting.getNumOfCategories() == 2)
        {
            ws = new Vec[1];
            bs = new double[1];
        }
        else
        {
            if(!(loss instanceof LossMC))
                throw new FailedToFitException("Loss function " + loss.getClass().getSimpleName() + " only supports binary classification");
            ws = new Vec[predicting.getNumOfCategories()];
            bs = new double[predicting.getNumOfCategories()];
        }
        setUpShared(numericAttributes);
    }
    
    @Override
    public void setUp(CategoricalData[] categoricalAttributes, int numericAttributes)
    {
        if(!(loss instanceof LossR))
            throw new FailedToFitException("Loss function " + loss.getClass().getSimpleName() + "does not support regression");
        
        ws = new Vec[1];
        bs = new double[1];
        setUpShared(numericAttributes);
    }
    
    private void setUpShared(int numericAttributes)
    {
        if(numericAttributes <= 0 )
            throw new FailedToFitException("LinearSGD requires numeric features to use");
        for(int i = 0; i < ws.length; i++)
            ws[i] = new ScaledVector(new DenseVector(numericAttributes));
        time = 0;
        l1U = 0;
        if(lambda1 > 0)
            l1Q = new double[ws.length][ws[0].length()];
        else
            l1Q = null;
    }

    @Override
    public void update(DataPoint dataPoint, int targetClass)
    {
        final double eta_t = decay.rate(time++, eta);
        Vec x = dataPoint.getNumericalValues();
        
        
        applyL2Reg(eta_t);
        
        //apply gradient updates
        if(ws.length == 1)
        {
            final double y = targetClass*2-1;
            final double lossD = ((LossC)loss).getDeriv(ws[0].dot(x)+bs[0], y);
            ws[0].mutableSubtract(eta_t*lossD, x);
            if(useBias)
                bs[0] -= eta_t*lossD;
        }
        else
        {
            Vec pred = new DenseVector(ws.length);
            for(int i = 0; i < ws.length; i++)
                pred.set(i, ws[i].dot(x)+bs[i]);
            ((LossMC)loss).process(pred, pred);
            ((LossMC)loss).deriv(pred, pred, targetClass);
            for(IndexValue iv : pred)
            {
                final int i = iv.getIndex();
                final double lossD = iv.getValue();
                ws[i].mutableSubtract(eta_t*lossD, x);
                if(useBias)
                    bs[i] -= eta_t*lossD;
            }
        }
        
        applyL1Reg(eta_t, x);
    }
    
    @Override
    public void update(DataPoint dataPoint, double targetValue)
    {
        final double eta_t = decay.rate(time++, eta);
        Vec x = dataPoint.getNumericalValues();
        
        applyL2Reg(eta_t);
        
        final double lossD = ((LossR)loss).getDeriv(ws[0].dot(x)+bs[0], targetValue);
        
        ws[0].mutableSubtract(eta_t*lossD, x);
        bs[0] -= eta_t*lossD;
        
        applyL1Reg(eta_t, x);
    }
    
    @Override
    public CategoricalResults classify(DataPoint data)
    {
        Vec x = data.getNumericalValues();
        if(ws.length == 1)
            return ((LossC)loss).getClassification(ws[0].dot(x)+bs[0]);
        else
        {
            Vec pred = new DenseVector(ws.length);
            for(int i = 0; i < ws.length; i++)
                pred.set(i, ws[i].dot(x)+bs[i]);
            ((LossMC)loss).process(pred, pred);
            return ((LossMC)loss).getClassification(pred);
        }
    }
    
    @Override
    public double regress(DataPoint data)
    {
        Vec x = data.getNumericalValues();
        return ((LossR)loss).getRegression(ws[0].dot(x)+bs[0]);
    }

    @Override
    public boolean supportsWeightedData()
    {
        return false;
    }

    /**
     * Applies L2 regularization to the model
     * @param eta_t the learning rate in use
     */
    private void applyL2Reg(final double eta_t)
    {
        if(lambda0 > 0)//apply L2 regularization
            for(Vec v : ws)
                v.mutableMultiply(1-eta_t*lambda0);
    }

    /**
     * Applies L1 regularization to the model
     * @param eta_t the learning rate in use
     * @param x the input vector the update is from
     */
    private void applyL1Reg(final double eta_t, Vec x)
    {
        //apply l1 regularization
        if(lambda1 > 0)
        {
            l1U += eta_t*lambda1;//line 6: in Tsuruoka et al paper, figure 2
            for(int k = 0; k < ws.length; k++)
            {
                final Vec w_k = ws[k];
                final double[] l1Q_k = l1Q[k];
                for(IndexValue iv : x)
                {
                    final int i = iv.getIndex();
                    //see "APPLYPENALTY(i)" on line 15: from Figure 2 in Tsuruoka et al paper
                    final double z = w_k.get(i);
                    double newW_i;
                    if (z > 0)
                        newW_i = Math.max(0, z - (l1U + l1Q_k[i]));
                    else
                        newW_i = Math.min(0, z + (l1U - l1Q_k[i]));
                    l1Q_k[i] += (newW_i - z);
                    w_k.set(i, newW_i);
                }
            }
        }
    }

    @Override
    public void train(RegressionDataSet dataSet, ExecutorService threadPool)
    {
        train(dataSet);
    }

    @Override
    public void train(RegressionDataSet dataSet)
    {
        BaseUpdateableRegressor.trainEpochs(dataSet, this, getEpochs());
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

    @Override
    public Vec getRawWeight(int index)
    {
        return ws[index];
    }

    @Override
    public double getBias(int index)
    {
        return bs[index];
    }

    @Override
    public int numWeightsVecs()
    {
        return ws.length;
    }
    
}
