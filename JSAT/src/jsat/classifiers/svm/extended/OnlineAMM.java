package jsat.classifiers.svm.extended;

import java.util.*;
import jsat.DataSet;
import jsat.classifiers.BaseUpdateableClassifier;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.CategoricalResults;
import jsat.classifiers.DataPoint;
import jsat.distributions.Distribution;
import jsat.distributions.Gamma;
import jsat.distributions.LogUniform;
import jsat.exceptions.FailedToFitException;
import jsat.linear.DenseVector;
import jsat.linear.ScaledVector;
import jsat.linear.Vec;
import jsat.linear.VecWithNorm;
import jsat.parameters.Parameter;
import jsat.parameters.Parameterized;
import jsat.utils.DoubleList;
import jsat.utils.IndexTable;
import jsat.utils.IntList;

/**
 * This is the Online variant of the Adaptive Multi-Hyperplane Machine (AMM) 
 * algorithm. It is related to linear SVMs where instead of having only a single
 * weight vector, it is extended to multi-class problems by giving each class 
 * its own weight vector. It is further extended by allowing each class to 
 * dynamically add new weight vectors to increase the non-linearity of the 
 * solution. <br>
 * This algorithm works best for problems with a very large number of data 
 * points where traditional kernelized SVMs are prohibitively expensive to train
 * due to computational cost. <br>
 * <br>
 * Unlike the batch variant, the online AMM algorithm has no convergence 
 * guarantees. However it still produces good results. 
 * <br>
 * See: 
 * <ul>
 * <li>Wang, Z., Djuric, N., Crammer, K., &amp; Vucetic, S. (2011). <i>Trading 
 * representability for scalability Adaptive Multi-Hyperplane Machine for 
 * nonlinear Classification</i>. In Proceedings of the 17th ACM SIGKDD 
 * international conference on Knowledge discovery and data mining - KDD ’11 
 * (p. 24). New York, New York, USA: ACM Press. doi:10.1145/2020408.2020420</li>
 * <li>Djuric, N., Lan, L., Vucetic, S., &amp; Wang, Z. (2014). <i>BudgetedSVM: A 
 * Toolbox for Scalable SVM Approximations</i>. Journal of Machine Learning 
 * Research, 14, 3813–3817. Retrieved from 
 * <a href="http://jmlr.org/papers/v14/djuric13a.html">here</a></li>
 * </ul>
 * 
 * @author Edward Raff
 */
public class OnlineAMM extends BaseUpdateableClassifier implements Parameterized
{

	private static final long serialVersionUID = 8291068484917637037L;
	/*
     * b/c of the batch learner we use a map, so that we dont have to think 
     * about how to handle re-assignment of data points to weight vectors. 
     * Also allows us to handle removed cases by checking if our owner is in the
     * map. Use nextID to make sure we give every new vec a unique ID with 
     * respect to the class label
     */
    protected List<Map<Integer, Vec>> weightMatrix;
    protected int[] nextID;
    protected double lambda;
    protected int k;
    protected double c;
    protected int time;
    protected int classBudget;
    
    /**
     * The default {@link #setPruneFrequency(int) frequency for pruning} is 
     * {@value #DEFAULT_PRUNE_FREQUENCY}.
     */
    public static final int DEFAULT_PRUNE_FREQUENCY = 10000;
    /**
     * The default {@link #setC(double)  pruning constant } is 
     * {@value #DEFAULT_PRUNE_CONSTANT}.
     */
    public static final double DEFAULT_PRUNE_CONSTANT = 10.0;
    /**
     * The default {@link #setClassBudget(int) class budget} is 
     * {@value #DEFAULT_CLASS_BUDGET}.
     */
    public static final int DEFAULT_CLASS_BUDGET = 50;
    /**
     * The default {@link #setLambda(double) regularization value} is 
     * {@value #DEFAULT_REGULARIZER}.
     */
    public static final double DEFAULT_REGULARIZER = 1e-2;

    /**
     * Creates a new online AMM learner
     */
    public OnlineAMM()
    {
        this(DEFAULT_REGULARIZER);
    }
    
    /**
     * Creates a new online AMM learner
     * @param lambda the regularization value to use
     */
    public OnlineAMM(double lambda)
    {
        this(lambda, DEFAULT_CLASS_BUDGET);
    }
    
    /**
     * Creates a new online AMM learner
     * @param lambda the regularization value to use
     * @param classBudget the maximum number of weight vectors for each class
     */
    public OnlineAMM(double lambda, int classBudget)
    {
        setLambda(lambda);
        setClassBudget(classBudget);
        setPruneFrequency(DEFAULT_PRUNE_FREQUENCY);
        setC(DEFAULT_PRUNE_CONSTANT);
    }

    /**
     * Copy constructor
     * @param toCopy the object to copy
     */
    public OnlineAMM(OnlineAMM toCopy)
    {
        if(toCopy.weightMatrix != null)
        {
            this.weightMatrix = new ArrayList<Map<Integer, Vec>>(toCopy.weightMatrix.size());
            for(Map<Integer, Vec> oldW : toCopy.weightMatrix)
            {
                Map<Integer, Vec> newW = new LinkedHashMap<Integer, Vec>(oldW.size());
                for(Map.Entry<Integer, Vec> entry : oldW.entrySet())
                    newW.put(entry.getKey(), entry.getValue().clone());
                this.weightMatrix.add(newW);
            }
            this.nextID = Arrays.copyOf(toCopy.nextID, toCopy.nextID.length);
        }
        this.time = toCopy.time;
        this.lambda = toCopy.lambda;
        this.k = toCopy.k;
        this.c = toCopy.c;
        this.classBudget = toCopy.classBudget;
        this.setEpochs(toCopy.getEpochs());
    }
    

    @Override
    public OnlineAMM clone()
    {
        return new OnlineAMM(this);
    }
    
    /**
     * Sets the regularization parameter for this algorithm. The original paper
     * suggests trying values 10<sup>-2</sup>, 10<sup>-3</sup>, ..., 
     * 10<sup>-6</sup>, 10<sup>-7</sup>. 
     * 
     * @param lambda the positive regularization parameter in (0, &infin;)
     */
    public void setLambda(double lambda)
    {
        if(lambda <= 0 || Double.isNaN(lambda) || Double.isInfinite(lambda))
            throw new IllegalArgumentException("Lambda must be positive, not " + lambda);
        this.lambda = lambda;
    }

    /**
     * Returns the regularization parameter
     * @return the regularization parameter
     */
    public double getLambda()
    {
        return lambda;
    }
    
    /**
     * Sets the frequency at which the weight vectors are pruned. Increasing the
     * frequency increases the aggressiveness of pruning. 
     * 
     * @param frequency the number of iterations between each pruning  
     */
    public void setPruneFrequency(int frequency )
    {
        if(frequency < 1)
            throw new IllegalArgumentException("Pruning frequency must be positive, not " + frequency);
        this.k = frequency;
    }
    
    /**
     * Returns the number of iterations between each pruning
     * @return the number of iterations between each pruning
     */
    public int getPruneFrequency()
    {
        return k;
    }

    /**
     * Sets the pruning constant which controls how powerful pruning is when 
     * pruning occurs. Increasing C increases how many weights will be pruned.
     * Changes to the scaling of feature vectors may require a change in the 
     * value of C
     * <br>
     * <b>NOTE:</b> This parameter <i>is not the same</i> as the standard C 
     * parameter associated with SVMs. 
     * @param c the positive pruning constant to use in (0, &infin;)
     */
    public void setC(double c)
    {
        if(c <= 0 || Double.isNaN(c) || Double.isInfinite(c))
            throw new IllegalArgumentException("C must be positive, not " + c);
        this.c = c;
    }

    /**
     * Returns the pruning constant
     * @return the pruning constant
     */
    public double getC()
    {
        return c;
    }

    /**
     * When given bad parameters there is the possibility for unbounded growth 
     * in the number of hyperplanes used. By setting this value to a reasonable 
     * upperbound catastrophic memory and CPU use can be avoided. 
     * @param classBudget the maximum number of hyperplanes allowed per class
     */
    public void setClassBudget(int classBudget)
    {
        if(classBudget < 1)
            throw new IllegalArgumentException("Number of hyperplanes must be positive, not " + classBudget);
        this.classBudget = classBudget;
    }

    /**
     * Returns the maximum number of hyperplanes allowed per class
     * @return the maximum number of hyperplanes allowed per class
     */
    public int getClassBudget()
    {
        return classBudget;
    }
    
    @Override
    public void setUp(CategoricalData[] categoricalAttributes, int numericAttributes, CategoricalData predicting)
    {
        if(numericAttributes < 1)
            throw new FailedToFitException("OnlineAMM requires numeric features to perform classification");
        weightMatrix = new ArrayList<Map<Integer, Vec>>(predicting.getNumOfCategories());
        for(int i = 0; i < predicting.getNumOfCategories(); i++)
            weightMatrix.add(new LinkedHashMap<Integer, Vec>());
        nextID = new int[weightMatrix.size()];
        time = 1;
    }

    @Override
    public void update(DataPoint dataPoint, final int y_t)
    {
        update(dataPoint, y_t, Integer.MIN_VALUE);
    }
    
    /**
     * Performs the work for an update. It can be used for regular online 
     * learning, or in the batch scenario where the assignments should not be 
     * updated with each online update. <br>
     * The output will change under certain circumstances. <br>
     * NOTE: this method may change in the future, dont rely on it
     * 
     * @param dataPoint the data point to use in the update
     * @param y_t the true label of the data point
     * @param z_t the hyperplane of the true class with the maximum response, or {@link Integer#MIN_VALUE} if it should be calculated
     * @return the index of the hyperplane of the true class with that maximum response
     */
    protected int update(DataPoint dataPoint, final int y_t, int z_t)
    {
        //2: (x_t, y_t) ← t-th example from S;
        final Vec x_t = dataPoint.getNumericalValues();
        
        //3: calculate z_t by (10);
        /*
         * Note, we use the same code for both online and batch AMM. If the 
         * input is the miniumum value, we are doing a normal online update.
         * If not, we use the given input & need to find out the response for 
         * the specified z_t instead
         * 
         * Not clear in the paper, how do we handle the case when z_t was 
         * assigned to someone removed? Lets just treat it as unknown. Should be
         * rare, so we wont worry about values changing and being reassigned. 
         * Especially since weight vectors near the front should be stable (they
         * survided longer than those formerly infront of them after all). 
         */
        double z_t_val;
        if(z_t == Integer.MIN_VALUE || z_t > nextID[y_t])//z_t is not known, so we will "update" it ourselves
        {
            z_t_val = 0.0;//infinte implicit zero weight vectors, so max is always at least 0
            z_t = -1;//negative value used to indicate the implicit was largest
            Map<Integer, Vec> w_yt = weightMatrix.get(y_t);
            for(Map.Entry<Integer, Vec> entry_yt : w_yt.entrySet())
            {
                Vec v = entry_yt.getValue();
                double tmp = x_t.dot(v);
                if(tmp >= z_t_val)
                {
                    z_t = entry_yt.getKey();
                    z_t_val = tmp;
                }
            }
        }
        else//z_t is given, we just need z_t_val
        {
            if(!weightMatrix.get(y_t).containsKey(z_t))
            {
                //happens if we were owned by a vec that has been removed
                return update(dataPoint, y_t, Integer.MIN_VALUE);//restart and have a new assignment given
            }
            if(z_t == -1)
                z_t_val = 0.0;//again, implicit
            else
                z_t_val = weightMatrix.get(y_t).get(z_t).dot(x_t);
        }
        
        //4: update W(++t) by (11)
        
        final double eta = 1.0/(lambda*time++);
        
        //computing i_t and j_t from equation (13)
        int i_t = (y_t > 0 ? 0 : 1);//j_t may be implicit, but i_t needs to belong to someone in the event of a tie. So just give it to the first class that isn't y_t
        double i_t_val = 0.0;
        int j_t = -1;
        
        for(int k = 0; k < weightMatrix.size(); k++)
        {
            if(k == y_t)
                continue;
            Map<Integer, Vec> w_k = weightMatrix.get(k);
            for(Map.Entry<Integer, Vec> entry_kj : w_k.entrySet())
            {
                Vec w_kj = entry_kj.getValue();
                double tmp = x_t.dot(w_kj);
                if(tmp > i_t_val)
                {
                    i_t = k;
                    j_t = entry_kj.getKey();
                    i_t_val = tmp;
                }
            }
        }
        //We need to check if the loss was greater than 0
        boolean nonZeroLoss = 0 < 1+i_t_val-z_t_val;
        
        //Now shrink all weights
        for(int i = 0; i < weightMatrix.size(); i++)
        {
            Map<Integer, Vec> w_i = weightMatrix.get(i);
            for(Map.Entry<Integer, Vec> w_entry_ij : w_i.entrySet())
            {
                int j = w_entry_ij.getKey();
                Vec w_ij = w_entry_ij.getValue();
                w_ij.mutableMultiply(-(eta*lambda-1));
                if(i == i_t && j == j_t && nonZeroLoss)
                    w_ij.mutableSubtract(eta, x_t);
                else if(i == y_t && j == z_t && nonZeroLoss)
                    w_ij.mutableAdd(eta, x_t);
            }
            //Also must check for implicit weight vectors needing an update (making them non-implicit)
            if (i == i_t && j_t == -1 && nonZeroLoss && w_i.size() < classBudget)
            {
                double norm = x_t.pNorm(2);
                Vec v = new DenseVector(x_t);
                v = new VecWithNorm(v, norm);
                v = new ScaledVector(v);
                v.mutableMultiply(-eta);
                w_i.put(nextID[i]++, v);
            }
            else if (i == y_t && z_t == -1 && nonZeroLoss && w_i.size() < classBudget)
            {
                double norm = x_t.pNorm(2);
                Vec v = new DenseVector(x_t);
                v = new VecWithNorm(v, norm);
                v = new ScaledVector(v);
                v.mutableMultiply(eta);
                w_i.put(nextID[i]++, v);
                //update z_t to point to the added value so we can return it correctly
                z_t = w_i.size()-1;
            }
        }
        
        if(time % k == 0)//Pruning time!
        {
            double threshold = c/((time-1)*lambda);
            
            IntList classOwner = new IntList(weightMatrix.size());
            IntList vecID = new IntList(weightMatrix.size());
            DoubleList normVal = new DoubleList(weightMatrix.size());
            for(int i = 0; i < weightMatrix.size(); i++)
            {
                for(Map.Entry<Integer, Vec> entry : weightMatrix.get(i).entrySet())
                {
                    Vec v = entry.getValue();
                    classOwner.add(i);
                    vecID.add(entry.getKey());
                    normVal.add(v.dot(v));
                }
            }
            
            IndexTable it = new IndexTable(normVal);
            
            for(int orderIndx = 0; orderIndx < normVal.size(); orderIndx++)
            {
                int i = it.index(orderIndx);
                double norm = normVal.get(i);
                if(norm >= threshold)
                    break;
                threshold -= norm;
                int classOf = classOwner.getI(i);
                weightMatrix.get(classOf).remove(vecID.getI(i));
            }
        }
        
        return z_t;
    }

    @Override
    public CategoricalResults classify(DataPoint data)
    {
        Vec x = data.getNumericalValues();
        int k_indx = 0;
        double maxVal = Double.NEGATIVE_INFINITY;
        for(int k = 0; k < weightMatrix.size(); k++)
        {
            for(Vec w_kj : weightMatrix.get(k).values())
            {
                double tmp = x.dot(w_kj);
                if(tmp > maxVal)
                {
                    k_indx = k;
                    maxVal = tmp;
                }
            }
        }
        
        CategoricalResults cr = new CategoricalResults(weightMatrix.size());
        cr.setProb(k_indx, 1.0);
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
     * {@link #setLambda(double) &lambda; } in AMM.
     *
     * @param d the data set to get the guess for
     * @return the guess for the &lambda; parameter
     */
    public static Distribution guessLambda(DataSet d)
    {
        return new LogUniform(1e-7, 1e-2);
    }
}
