
package jsat.classifiers.bayesian.graphicalmodel;

import java.util.Set;

import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.DataPoint;
import jsat.utils.IntList;
import jsat.utils.IntSet;
import jsat.utils.ListUtils;
import static java.lang.Math.*;
import static jsat.math.SpecialMath.*;

/**
 * An implementation of the K2 algorithm for learning the structure of a Bayesian Network. When trained, 
 * if no network has been specified, the K2 algorithm will attempt to learn a network structure. The 
 * network structure can also be learned by calling {@link #learnNetwork(jsat.classifiers.ClassificationDataSet) } directly. 
 * <br><br>
 * Note, that the K2 algorithm attempts to learn a whole network structure, and may learn things that are relevant 
 * for the classification task. K2 often does not provide satisfactory results for classification. 
 * <br><br>
 * See: <i>A bayesian method for the induction of probabilistic networks from data</i>. Gregory F. Cooper and Edward Herskovits. 
 * @author Edward Raff
 */
public class K2NetworkLearner extends DiscreteBayesNetwork
{


	private static final long serialVersionUID = -9681177007308829L;

	public K2NetworkLearner()
    {
        super();
    }
    
    /**
     * list of all possible values of the attribute xi
     */
    private int[] ri;
    private int maxParents;

    /**
     * Sets the maximum number of parents to allow a node when learning the network structure. If a non zero value is supplied, nodes will be allowed any number of parents. 
     * @param maxParents sets the maximum number of parents a node may learn
     */
    public void setMaxParents(int maxParents)
    {
        this.maxParents = maxParents;
    }

    /**
     * Returns the maximum number of parents allowed when learning a network structure, or zero if any number of parents are valid. 
     * @return the maximum number of parents a node man learn
     */
    public int getMaxParents()
    {
        return max(maxParents, 0);
    }
    
    /**
     * Learns the network structure from the given data set. 
     * @param D the data set to learn the network from 
     */
    public void learnNetwork(ClassificationDataSet D)
    {
        IntList varOrder = new IntList(D.getNumCategoricalVars()+1);
        varOrder.add(D.getNumCategoricalVars());//Classification target will be evaluated first
        ListUtils.addRange(varOrder, 0, D.getNumCategoricalVars(), 1);
        ri = new int[varOrder.size()];
        for(int i : varOrder)
            if(i == D.getNumCategoricalVars())
                ri[i] = D.getClassSize();
            else
                ri[i] = D.getCategories()[i].getNumOfCategories();
        
        int u = maxParents;
        if(u <= 0)
            u = ri.length;
        
        /**
         * Stores the set of variables preceding the current one being evaluated
         */
        Set<Integer> preceding = new IntSet();
        for(int i : varOrder)//Loop of the variables in the intended order
        {
            Set<Integer> pi = new IntSet();//The current parrents of variable i
            double pOld = f(i, pi, D);
            boolean OKToProceed = true;
            Set<Integer> candidates = new IntSet(preceding);
            while(OKToProceed && pi.size() < u)
            {
                if(candidates.isEmpty())
                    break;//Break out of the loop, no candidates left. 
                
                //Best candidate solution
                double pNew = Double.NEGATIVE_INFINITY;
                //The best candidate
                int z = -1;
                candidates.removeAll(pi);
                //Find the variable that maximizes our gain 
                for(int candidate : candidates)
                {
                    pi.add(candidate);
                    double tmp = f(i, pi, D);
                    if(tmp > pNew)
                    {
                        pNew = tmp;
                        z = candidate;
                    }
                    pi.remove(candidate);
                }
                
                if(pNew > pOld)
                {
                    pOld = pNew;
                    pi.add(z);
                }
                else
                    OKToProceed = false;
            }
            
            for(int parrent : pi)
                depends(parrent, i);
            
            preceding.add(i);
        }
        
        ri = null;
    }

    @Override
    public void trainC(ClassificationDataSet dataSet)
    {
        if(dag.getNodes().isEmpty() || dag.getParents(dataSet.getNumCategoricalVars()).isEmpty())
            learnNetwork(dataSet);
            
        super.trainC(dataSet);
    }
    
    
    
    /**
     * Queries the data set for the number of instances that have each possible combination of values.
     * <tt>classes</tT> and <tt>values</tt> should have the same length. Each value in classes 
     * corresponds to the target value specified in <tt>values</tt>. We will return the number 
     * of data points that satisfy all class value pairs
     * 
     * 
     * @param classes the classes to check
     * @param values the values to check for
     * @param D the data set to search
     * @return the number of times the value constraints are satisfied in the data set
     */
    private double query(int[] classes, int[] values, ClassificationDataSet D)
    {
        double count = 1;
        
        for(int i = 0; i < D.getSampleSize(); i++)
        {
            DataPoint dp = D.getDataPoint(i);
            //Use j to break early (set value) or indicate success (j == classes.length)
            int j;
            for(j = 0; j < classes.length; j++)
            {
                if(classes[j] == D.getNumCategoricalVars())//Special case
                {
                    if(D.getDataPointCategory(i) != values[j])
                        j = classes.length+1;
                }
                else if(dp.getCategoricalValue(j) != values[j])
                    j = classes.length+1;
            }
            
            if(j == classes.length)
                count+=dp.getWeight();
        }
        
        return count;
    }
    
    public double f(int i, Set<Integer> pi, ClassificationDataSet D)
    {
        double term2 = 0.0;
        double Nijk = 0.0;
        
        if(pi.isEmpty())//Special case
        {
            int[] classes = new int[] {i};
            int[] values = new int[1];
            for(int k = 0; k < ri[i]; k++)
            {
                values[0] = k;
                double count = query(classes, values, D);
                Nijk += count;
                term2 += lnGamma(count+1);
            }
            
            return ((lnGamma(ri[i]) - lnGamma(Nijk + ri[i])) + term2);
        }
        
        double fullProduct = 0.0;
        //General case
        int[] classes = new int[pi.size()+1];
        int[] values = new int[pi.size()+1];
        int c = 0;
        for(int clas : pi)
            classes[c++] = clas;
        classes[c] = i;//Last one is the one we are currently evaluating
        //Default all values to zero, which is fine
        
        //We need to compute the sum for ever possible combination of class values. 
        //We do this by incrementing the values array and breaking out when we get to an invalid state
        while(true)
        {
            term2 = Nijk = 0.0;
            
            for(int k = 0; k < ri[i]; k++)
            {
                values[pi.size()] = k;
                double count = query(classes, values, D);
                Nijk += count;
                term2 += lnGamma(count+1);
            }
            
            fullProduct += (lnGamma(ri[i]) - lnGamma(Nijk + ri[i])) + term2;
            
            //Increment the variable count
            int pos = 0;
            values[pos]++;
            values[pi.size()] = 0;//We set this to zero. If its value is non zero after this loop, 
            //then all values have been iterated past their max value, they have rolled 
            //over to all zeros, and we are done
            
            while(values[pos] >= ri[classes[pos]] && pos < pi.size())
            {
                values[pos++] = 0;
                values[pos]++;
            }
            if(values[pi.size()] != 0)
                break;
        }
        
        return (fullProduct);
    }
    
}

