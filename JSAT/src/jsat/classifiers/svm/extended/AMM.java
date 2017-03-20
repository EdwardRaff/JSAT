package jsat.classifiers.svm.extended;

import java.util.*;
import java.util.concurrent.ExecutorService;
import jsat.classifiers.ClassificationDataSet;
import jsat.linear.Vec;
import jsat.utils.IntList;
import jsat.utils.ListUtils;
import jsat.utils.random.RandomUtil;
import jsat.utils.random.XORWOW;

/**
 * This is the batch variant of the Adaptive Multi-Hyperplane Machine (AMM) 
 * algorithm. It is related to linear SVMs where instead of having only a single
 * weight vector, it is extended to multi-class problems by giving each class 
 * its own weight vector. It is further extended by allowing each class to 
 * dynamically add new weight vectors to increase the non-linearity of the 
 * solution. <br>
 * This algorithm works best for problems with a very large number of data 
 * points where traditional kernelized SVMs are prohibitively expensive to train
 * due to computational cost. <br>
 * While the AMM trained in a batch setting can continue to be updated in an 
 * online fashion, the accuracy may reduce if done. This is because only the 
 * batch variant will reach a local optima.<br>
 * For this version the {@link #setEpochs(int) } method controls the total 
 * number of iterations of the learning algorithm. A small value in [5, 20] 
 * should be sufficient. 
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
public class AMM extends OnlineAMM
{

	private static final long serialVersionUID = -9198419566231617395L;
	private int subEpochs = 1;

    /**
     * Creates a new batch AMM learner
     */
    public AMM()
    {
        this(DEFAULT_REGULARIZER);
    }
    
    /**
     * Creates a new batch AMM learner
     * @param lambda the regularization value to use
     */
    public AMM(double lambda)
    {
        this(lambda, DEFAULT_CLASS_BUDGET);
    }
    
    /**
     * Creates a new batch AMM learner
     * @param lambda the regularization value to use
     * @param classBudget the maximum number of weight vectors for each class
     */
    public AMM(double lambda, int classBudget)
    {
        super(lambda, classBudget);
        setEpochs(10);
    }
    
    /**
     * Copy constructor
     * @param toCopy the object to copy
     */
    public AMM(AMM toCopy)
    {
        super(toCopy);
        this.subEpochs = toCopy.subEpochs;
    }

    /**
     * Each iteration of the batch AMM algorithm requires at least one epoch 
     * over the training set. This control how many epochs make up each
     * iteration of training. 
     * 
     * @param subEpochs the number passes through the training set done on each 
     * iteration of training
     */
    public void setSubEpochs(int subEpochs)
    {
        if(subEpochs < 1)
            throw new IllegalArgumentException("subEpochs must be positive, not " + subEpochs);
        this.subEpochs = subEpochs;
    }

    /**
     * Returns the number of passes through the data set done on each iteration
     * @return the number of passes through the data set done on each iteration
     */
    public int getSubEpochs()
    {
        return subEpochs;
    }

    @Override
    public void trainC(ClassificationDataSet dataSet, ExecutorService threadPool)
    {
        trainC(dataSet);
    }

    @Override
    public void trainC(ClassificationDataSet dataSet)
    {   
        IntList randOrder = new IntList(dataSet.getSampleSize());
        ListUtils.addRange(randOrder, 0, dataSet.getSampleSize(), 1);
        Random rand = RandomUtil.getRandom();
        
        int[] Z = new int[randOrder.size()];
        
        /*
         * For Algorithm 1, instead of a random assignment, we initialized z(1) 
         * by a single scan of data using Online AMM
         */
        setUp(dataSet.getCategories(), dataSet.getNumNumericalVars(), dataSet.getPredicting());
        Collections.shuffle(randOrder, rand);
        //also perform step 1: initialize z(1)
        for(int i : randOrder)
            Z[i] = update(dataSet.getDataPoint(i), dataSet.getDataPointCategory(i), Integer.MIN_VALUE);
        
        time = 1;//rest the time since we are "Starting" now, and before was just a better than random intial state
        
        int outerEpoch = 0;
        do//2: repeat
        {
            /* Solve each sub-problem P(W|z(r)): lines 4 ∼ 7*/ 
            for(int subEpoch = 0; subEpoch < subEpochs; subEpoch++)
            {
                Collections.shuffle(randOrder, rand);
                for(int i : randOrder)
                    Z[i] = update(dataSet.getDataPoint(i), dataSet.getDataPointCategory(i), Z[i]);//only changing value in certain valid cases
            }
            // 8: compute z(++r) using (9); /* Reassign z */
            int changed = 0;
            for(int i = 0; i < randOrder.size(); i++)
            {
                Vec x_t = dataSet.getDataPoint(i).getNumericalValues();
                double z_t_val = 0.0;//infinte implicit zero weight vectors, so max is always at least 0
                int z_t = -1;//negative value used to indicate the implicit was largest
                Map<Integer, Vec> w_yt = weightMatrix.get(dataSet.getDataPointCategory(i));
                for(Map.Entry<Integer, Vec> w_yt_entry : w_yt.entrySet())
                {
                    Vec v = w_yt_entry.getValue();
                    double tmp = x_t.dot(v);
                    if(tmp >= z_t_val)
                    {
                        z_t = w_yt_entry.getKey();
                        z_t_val = tmp;
                    }
                }
                
                if(Z[i] != z_t)
                {
                    changed++;
                    Z[i] = z_t;
                }
            }
            
            if(changed == 0)
                break;
        }
        while(++outerEpoch < getEpochs());
    }

    @Override
    public AMM clone()
    {
        return new AMM(this);
    }
}
