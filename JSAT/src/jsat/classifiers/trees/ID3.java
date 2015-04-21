
package jsat.classifiers.trees;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.logging.Level;
import java.util.logging.Logger;

import jsat.classifiers.CategoricalData;
import jsat.classifiers.CategoricalResults;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.Classifier;
import jsat.classifiers.DataPoint;
import jsat.classifiers.DataPointPair;
import jsat.utils.FakeExecutor;
import jsat.utils.IntSet;
import jsat.utils.ModifiableCountDownLatch;

/**
 *
 * @author Edward Raff
 */
public class ID3 implements Classifier
{
    

	private static final long serialVersionUID = -8473683139353205898L;
	private CategoricalData predicting;
    private CategoricalData[] attributes;
    private ID3Node root;
    private ModifiableCountDownLatch latch;

    public CategoricalResults classify(DataPoint data)
    {
        return walkTree(root, data);
    }
    
    static private CategoricalResults walkTree(ID3Node node, DataPoint data)
    {
        if(node.isLeaf())
            return node.getResult();
        
        return walkTree(node.getNode(data.getCategoricalValue(node.getAttributeId())), data);
    }

    public void trainC(ClassificationDataSet dataSet, ExecutorService threadPool)
    {
        if(dataSet.getNumNumericalVars() != 0)
            throw new RuntimeException("ID3 only supports categorical data");
        
        predicting = dataSet.getPredicting();
        attributes = dataSet.getCategories();
        List<DataPointPair<Integer>> dataPoints = dataSet.getAsDPPList();
        
        Set<Integer> availableAttributes = new IntSet(dataSet.getNumCategoricalVars());
        for(int i = 0; i < dataSet.getNumCategoricalVars(); i++)
            availableAttributes.add(i);
        latch = new ModifiableCountDownLatch(1);
        root = buildTree(dataPoints, availableAttributes, threadPool);    
        try
        {
            latch.await();
        }
        catch (InterruptedException ex)
        {
            Logger.getLogger(ID3.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    public void trainC(ClassificationDataSet dataSet)
    {
        trainC(dataSet, new FakeExecutor());      
    }
    
    private ID3Node buildTree( List<DataPointPair<Integer>> dataPoints, Set<Integer> remainingAtribues, final ExecutorService threadPool)
    {
        double curEntropy = entropy(dataPoints);
        double size = dataPoints.size();
        
        if(remainingAtribues.isEmpty() || curEntropy == 0)
        {
            CategoricalResults cr = new CategoricalResults(predicting.getNumOfCategories());
            for(DataPointPair<Integer> dpp : dataPoints)
                cr.setProb(dpp.getPair(), cr.getProb(dpp.getPair()) + 1);
            cr.divideConst(size);

            latch.countDown();
            return new ID3Node(cr);
        }
        
        int bestAttribute = -1;
        double bestInfoGain = Double.MIN_VALUE;
        List<List<DataPointPair<Integer>>> bestSplit = null;
        
        for(int attribute : remainingAtribues)
        {
            List<List<DataPointPair<Integer>>> newSplit = new ArrayList<List<DataPointPair<Integer>>>(attributes[attribute].getNumOfCategories());
            for( int i = 0; i < attributes[attribute].getNumOfCategories(); i++)
                newSplit.add( new ArrayList<DataPointPair<Integer>>());
            
            //Putting the datapoints in their respective bins by attribute value
            for(DataPointPair<Integer> dpp : dataPoints)
                newSplit.get(dpp.getDataPoint().getCategoricalValue(attribute)).add(dpp);
            
            double splitEntrop = 0;
            for(int i = 0; i < newSplit.size(); i++)
                splitEntrop += entropy(newSplit.get(i))*newSplit.get(i).size()/size;
            
            double infoGain = curEntropy - splitEntrop;
            if(infoGain > bestInfoGain)
            {
                bestAttribute = attribute;
                bestInfoGain = infoGain;
                bestSplit = newSplit;
            }
                
        }
        
        final ID3Node node = new ID3Node(attributes[bestAttribute].getNumOfCategories(), bestAttribute);
        final Set<Integer> newRemaining = new IntSet(remainingAtribues);
        newRemaining.remove(bestAttribute);
        for(int i = 0; i < bestSplit.size(); i++)
        {
            final int ii = i;
            final List<DataPointPair<Integer>> bestSplitII = bestSplit.get(ii);
            latch.countUp();
            threadPool.submit(new Runnable() {

                public void run()
                {
                    node.setNode(ii, buildTree(bestSplitII, newRemaining, threadPool));
                }
            });
            
        }
        
        latch.countDown();
        return node;
    }
    
    static private class ID3Node
    {
        ID3Node[] children;
        CategoricalResults cr;
        int attributeId;

        private ID3Node()
        {
        }

        /**
         * Constructs a parent
         * @param atributes the number of possible values for the attribute this node should split on
         */
        public ID3Node(int atributes, int attributeId)
        {
            cr = null;
            children = new ID3Node[atributes];
            this.attributeId = attributeId;
        }

        /**
         * Constructs a leaf
         * @param cr the result to return for reaching this leaf node
         */
        public ID3Node( CategoricalResults cr)
        {
            this.children = null;
            this.cr = cr;
        }
        
        public boolean isLeaf()
        {
            return cr != null;
        }
        
        public void setNode(int i, ID3Node node)
        {
            children[i] = node;
        }
        
        public ID3Node getNode(int i)
        {
            return children[i];
        }

        public int getAttributeId()
        {
            return attributeId;
        }

        public CategoricalResults getResult()
        {
            return cr;
        }
        
        public ID3Node copy()
        {
            ID3Node copy = new ID3Node();
            copy.cr = this.cr;
            copy.attributeId = this.attributeId;
            if(this.children != null)
            {
                copy.children = new ID3Node[this.children.length];
                for(int i = 0; i < children.length; i++)
                    copy.children[i] = this.children[i].copy();
                    
            }
            return copy;
        }
        
    }
    
    
    private double entropy( List<DataPointPair<Integer>> s)
    {
        if(s.isEmpty())
            return 0;
        double[] probs = new double[predicting.getNumOfCategories()];
        for(DataPointPair<Integer> dpp : s)
            probs[dpp.getPair()] += 1;
        for(int i = 0; i < probs.length; i++)
            probs[i] /= s.size();
        
        double entr = 0;
        
        for(int i = 0; i < probs.length; i++)
            if(probs[i] != 0)
                entr += probs[i] * (Math.log(probs[i])/Math.log(2));
        //The entr will be negative unless it is zero, this way we dont return negative zero
        return Math.abs(entr);
    }
    

    public boolean supportsWeightedData()
    {
        return false;
    }

    public Classifier clone()
    {
        ID3 copy = new ID3();
        
        copy.attributes = this.attributes;
        copy.latch = null;
        copy.predicting = this.predicting;
        copy.root = this.root.copy();
        
        return copy;
    }
    
}
