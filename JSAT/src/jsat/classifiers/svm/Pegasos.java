package jsat.classifiers.svm;

import java.util.*;
import java.util.concurrent.ExecutorService;
import jsat.DataSet;
import jsat.SingleWeightVectorModel;
import jsat.classifiers.*;
import jsat.classifiers.calibration.BinaryScoreClassifier;
import jsat.distributions.Distribution;
import jsat.distributions.Gamma;
import jsat.distributions.LogUniform;
import jsat.exceptions.FailedToFitException;
import jsat.linear.*;
import jsat.parameters.Parameter;
import jsat.parameters.Parameterized;
import jsat.utils.IntList;
import jsat.utils.ListUtils;

/**
 * Implements the linear kernel mini-batch version of the Pegasos SVM 
 * classifier. It performs updates stochastically and is very fast. <br>
 * Because Pegasos updates the primal directly, there are no support vectors 
 * saved from the training set.
 * <br><br>
 * See: Shalev-Shwartz, S., Singer, Y.,&amp;Srebro, N. (2007). <i>Pegasos : Primal
 * Estimated sub-GrAdient SOlver for SVM</i>. 24th international conference on 
 * Machine learning (pp. 807–814). New York, NY: ACM. 
 * doi:10.1145/1273496.1273598
 * 
 * @author Edward Raff
 */
public class Pegasos implements BinaryScoreClassifier, Parameterized, SingleWeightVectorModel
{

	private static final long serialVersionUID = -2145631476467081171L;
	private int epochs;
    private double reg;
    private int batchSize;
    private boolean projectionStep = false;
    private Vec w;
    private double bias;
    
    /**
     * The default number of epochs is {@value #DEFAULT_EPOCHS}
     */
    public static final int DEFAULT_EPOCHS = 5;
    /**
     * The default regularization value is {@value #DEFAULT_REG}
     */
    public static final double DEFAULT_REG = 1e-4;
    /**
     * The default batch size is {@value #DEFAULT_BATCH_SIZE}
     */
    public static final int DEFAULT_BATCH_SIZE = 1;

    /**
     * Creates a new Pegasos SVM classifier using default values. 
     */
    public Pegasos()
    {
        this(DEFAULT_EPOCHS, DEFAULT_REG, DEFAULT_BATCH_SIZE);
    }

    /**
     * Creates a new Pegasos SVM classifier
     * @param epochs the number of training iterations
     * @param reg the regularization term
     * @param batchSize the batch size 
     */
    public Pegasos(int epochs, double reg, int batchSize)
    {
        setEpochs(epochs);
        setRegularization(reg);
        setBatchSize(batchSize);
    }

    /**
     * Copy constructor
     * @param toCopy the object to copy
     */
    public Pegasos(Pegasos toCopy)
    {
        this.epochs = toCopy.epochs;
        this.reg = toCopy.reg;
        this.batchSize = toCopy.batchSize;
        if(toCopy.w != null)
            this.w = toCopy.w.clone();
        this.bias = toCopy.bias;
        this.projectionStep = toCopy.projectionStep;
    }
    
    /**
     * Sets the batch size used during training. At each epoch, a batch of 
     * randomly selected data points will be used to update. 
     * 
     * @param batchSize the number of data points to use when updating 
     */
    public void setBatchSize(int batchSize)
    {
        if(batchSize < 1)
            throw new ArithmeticException("At least one sample must be take at each iteration");
        this.batchSize = batchSize;
    }

    /**
     * Returns the number of points used in each iteration
     * @return the number of points used in each iteration
     */
    public int getBatchSize()
    {
        return batchSize;
    }

    /**
     * Sets the number of iterations through the training set that will be
     * performed. 
     * @param epochs the number of iterations
     */
    public void setEpochs(int epochs)
    {
        if(epochs < 1)
            throw new ArithmeticException("Must perform a positive number of epochs");
        this.epochs = epochs;
    }

    /**
     * Returns the number of iterations of updating that will be done
     * @return the number of iterations
     */
    public double getEpochs()
    {
        return epochs;
    }

    /**
     * Sets whether or not to use the projection step after each update per 
     * iteration
     * 
     * @param projectionStep whether or not to use the projection step
     */
    public void setProjectionStep(boolean projectionStep)
    {
        this.projectionStep = projectionStep;
    }

    /**
     * Returns whether or not the projection step is in use after each iteration
     * @return <tt>true</tt> if the projection step will be performed
     */
    public boolean isProjectionStep()
    {
        return projectionStep;
    }

    /**
     * Sets the regularization constant used for learning. The regularization 
     * must be positive, and the learning rate is proportional to the 
     * regularization value. This means regularizations very near zero will 
     * take a long time to converge. 
     * 
     * @param reg the regularization to apply
     */
    public void setRegularization(double reg)
    {
        if(Double.isInfinite(reg) || Double.isNaN(reg) || reg <= 0.0)
            throw new ArithmeticException("Pegasos requires a positive regularization cosntant");
        this.reg = reg;
    }

    /**
     * Returns the amount of regularization to used in training
     * @return the regularization parameter. 
     */
    public double getRegularization()
    {
        return reg;
    }

    @Override
    public Vec getRawWeight()
    {
        return w;
    }

    @Override
    public double getBias()
    {
        return bias;
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
    public Pegasos clone()
    {
        return new Pegasos(this);
    }

    @Override
    public CategoricalResults classify(DataPoint data)
    {
        CategoricalResults cr = new CategoricalResults(2);
        
        if(getScore(data) < 0)
            cr.setProb(0, 1.0);
        else
            cr.setProb(1, 1.0);
        
        return cr;
    }

    @Override
    public double getScore(DataPoint dp)
    {
        return w.dot(dp.getNumericalValues())+bias;
    }

    @Override
    public void trainC(ClassificationDataSet dataSet, ExecutorService threadPool)
    {
        trainC(dataSet);
    }

    @Override
    public void trainC(ClassificationDataSet dataSet)
    {
        if(dataSet.getClassSize() != 2)
            throw new FailedToFitException("SVM only supports binary classificaiton problems");
        final int m = dataSet.getSampleSize();
        w = new DenseVector(dataSet.getNumNumericalVars());
        if(projectionStep)
            w = new VecWithNorm(w, 0.0);
        w = new ScaledVector(w);
        bias = 0;
        
        
        IntList miniBatch = new IntList(batchSize);
        IntList randOrder = new IntList(m);
        ListUtils.addRange(randOrder, 0, m, 1);
        
        int t = 0;
        for (int epoch = 0; epoch < epochs; epoch++)//start at 1 for convinence
        {
            Collections.shuffle(randOrder);

            for (int indx = 0; indx < m; indx += batchSize)
            {
                t++;
                miniBatch.clear();
                miniBatch.addAll(randOrder.subList(indx, Math.min(indx+batchSize, m)));
                //Filter to only the points that have the correct label
                Iterator<Integer> iter = miniBatch.iterator();
                while (iter.hasNext())
                {
                    int i = iter.next();
                    if (getSign(dataSet, i) * (w.dot(getX(dataSet, i)) + bias) >= 1)
                        iter.remove();
                }

                final double nt = 1.0 / (reg * t);

                w.mutableMultiply(1.0 - nt * reg);

                for (int i : miniBatch)
                {
                    double sign = getSign(dataSet, i);
                    Vec x = getX(dataSet, i);
                    final double s = sign * nt /batchSize;
                    w.mutableAdd(s, x);
                    bias += s;
                }

                if (projectionStep)
                {
                    double norm = w.pNorm(2);
                    double mult = Math.min(1, 1.0 / (Math.sqrt(reg) * norm));
                    w.mutableMultiply(mult);
                    bias *= mult;
                }
            }
        }
    }

    @Override
    public boolean supportsWeightedData()
    {
        return false;
    }

    private Vec getX(ClassificationDataSet dataSet, int i)
    {
        return dataSet.getDataPoint(i).getNumericalValues();
    }

    private double getSign(ClassificationDataSet dataSet, int i)
    {
        return dataSet.getDataPointCategory(i) == 1 ? 1.0 : -1.0;
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
     * {@link #setRegularization(double) } in Pegasos.
     *
     * @param d the data set to get the guess for
     * @return the guess for the &lambda; parameter
     */
    public static Distribution guessRegularization(DataSet d)
    {
        return new LogUniform(1e-7, 1e-2);
    }
}
