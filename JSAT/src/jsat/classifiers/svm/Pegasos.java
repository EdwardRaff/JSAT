package jsat.classifiers.svm;

import java.util.*;
import java.util.concurrent.ExecutorService;
import jsat.classifiers.*;
import jsat.classifiers.calibration.BinaryScoreClassifier;
import jsat.exceptions.FailedToFitException;
import jsat.linear.*;
import jsat.parameters.Parameter;
import jsat.parameters.Parameterized;

/**
 * Implements the linear kernel mini-batch version of the Pegasos SVM 
 * classifier. It performs updates stochastically and is very fast. <br>
 * Because Pegasos updates the primal directly, there are no support vectors 
 * saved from the training set.
 * <br><br>
 * See: Shalev-Shwartz, S., Singer, Y., & Srebro, N. (2007). <i>Pegasos : Primal
 * Estimated sub-GrAdient SOlver for SVM</i>. 24th international conference on 
 * Machine learning (pp. 807–814). New York, NY: ACM. 
 * doi:10.1145/1273496.1273598
 * 
 * @author Edward Raff
 */
public class Pegasos implements BinaryScoreClassifier, Parameterized
{
    private double epochs;
    private double reg;
    private int batchSize;
    private boolean projectionStep = false;
    private Vec w;
    private double bias;
    
    /**
     * The default number of epochs is {@value #DEFAULT_EPOCHS}
     */
    public static final int DEFAULT_EPOCHS = 1000;
    /**
     * The default regularization value is {@value #DEFAULT_REG}
     */
    public static final double DEFAULT_REG = 1e-4;
    /**
     * The default batch size is {@value #DEFAULT_BATCH_SIZE}
     */
    public static final int DEFAULT_BATCH_SIZE = 1;
    
    private final List<Parameter> params = Collections.unmodifiableList(Parameter.getParamsFromMethods(this));
    private final Map<String, Parameter> paramMap = Parameter.toParameterMap(params);

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
    public Pegasos(double epochs, double reg, int batchSize)
    {
        setEpochs(epochs);
        setRegularization(reg);
        setBatchSize(batchSize);
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
     * Sets the number of iterations of training that will be performed. 
     * @param epochs the number of iterations
     */
    public void setEpochs(double epochs)
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
    public Pegasos clone()
    {
        Pegasos clone = new Pegasos(epochs, reg, batchSize);
        if(this.w != null)
            clone.w = this.w.clone();
        return clone;
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
        /**
         * Scale variable
         */
        double scale = 1;
        /**
         * Current norm^2
         */
        double v = 0;
        
        Random rand = new Random();
        final Set<Integer> miniBatch = new HashSet<Integer>(batchSize*2);
        
        for(int t = 1; t <= epochs; t++)//start at 1 for convinence
        {
            miniBatch.clear();
            while(miniBatch.size() < batchSize)
                miniBatch.add(rand.nextInt(m));
            //Filter to only the points that have the correct label
            Iterator<Integer> iter = miniBatch.iterator();
            while(iter.hasNext())
            {
                int i = iter.next();
                if(getSign(dataSet, i)*scale*(w.dot(getX(dataSet, i))+bias) >= 1)
                    iter.remove();
            }
                
            
            final double nt = 1.0/(reg*t);
            
            scale *= (1.0-nt*reg);
            v *= Math.pow((1.0-nt*reg), 2);
            if(scale == 0.0)
            {
                scale = 1.0;
                v = 0.0;
                w.zeroOut();
                bias = 0;
            }
            
            for(int i : miniBatch)
            {
                double sign = getSign(dataSet, i);
                Vec x = getX(dataSet, i);
                final double s = sign*nt/(batchSize*scale);
                //TODO update the norm in a more clever manner
                if(projectionStep)//update norm
                    for(IndexValue iv : x)
                        v -= Math.pow(scale*w.get(iv.getIndex()), 2);
                w.mutableAdd( s, x);
                bias += s;
                if(projectionStep)
                    for(IndexValue iv : x)
                        v += Math.pow(scale*w.get(iv.getIndex()), 2);
            }
            
            if(projectionStep)
            {
                double norm = Math.sqrt(v);
                double mult = Math.min(1, 1.0/(Math.sqrt(reg)*norm));
                if(mult != 1)
                {
                    //w.mutableMultiply(mult);
                    scale *= mult;
                    v *= Math.pow(mult, 2);
                    if(scale == 0.0)
                    {
                        scale = 1.0;
                        v = 0.0;
                        w.zeroOut();
                        bias = 0;
                    }
                }
            }
        }
        w.mutableMultiply(scale);
        bias *= scale;
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
        return params;
    }

    @Override
    public Parameter getParameter(String paramName)
    {
        return paramMap.get(paramName);
    }
}
