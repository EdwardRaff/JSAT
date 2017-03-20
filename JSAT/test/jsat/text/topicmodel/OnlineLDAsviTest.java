package jsat.text.topicmodel;

import jsat.text.topicmodel.OnlineLDAsvi;

import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import jsat.SimpleDataSet;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.DataPoint;
import jsat.distributions.multivariate.Dirichlet;
import jsat.linear.*;
import jsat.utils.IntSet;
import jsat.utils.SystemInfo;
import jsat.utils.random.RandomUtil;
import jsat.utils.random.XORWOW;

import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 *
 * @author Edward Raff
 */
public class OnlineLDAsviTest
{
    private static final int rows = 5;
    
    public OnlineLDAsviTest()
    {
    }
    
    @BeforeClass
    public static void setUpClass()
    {
    }
    
    @AfterClass
    public static void tearDownClass()
    {
    }
    
    @Before
    public void setUp()
    {
    }
    
    @After
    public void tearDown()
    {
    }

    /**
     * Test of model method, of class OnlineLDAsvi.
     */
    @Test
    public void testModel()
    {
        System.out.println("model");
        ExecutorService ex = Executors.newFixedThreadPool(SystemInfo.LogicalCores);
        
        for(int iters = 0; iters < 2; iters++)//controls whether parallel or single threaded verison is run
        {
            int attempts = 3;
            do
            {   
                //create the basis set to sample from
                List<Vec> basis = new ArrayList<Vec>();

                for(int i = 0; i < rows; i++)
                {
                    Vec b0 = new SparseVector(rows*rows);
                    for(int a = 0; a < rows; a++)
                        b0.set(i*5+a, 1.0);

                    Vec b1 = new SparseVector(rows*rows);
                    for(int a = 0; a < rows; a++)
                        b1.set(a*rows+i, 1.0);

                    b0.mutableDivide(b0.sum());
                    b1.mutableDivide(b1.sum());
                    basis.add(b0);
                    basis.add(b1);
                }

                //create the training set
                double alpha = 0.1;
                List<DataPoint> docs = new ArrayList<DataPoint>();
                Dirichlet dirichlet = new Dirichlet(new ConstantVector(alpha, basis.size()));
                Random rand = RandomUtil.getRandom();
                for(Vec topicSample : dirichlet.sample(100000, rand))
                {
                    Vec doc = new DenseVector(basis.get(0).length());
                    //sample 40 times
                    for(int i = 0; i < 100; i++)
                    {
                        double topicRand = rand.nextDouble();
                        int topic = 0;
                        double sum = topicSample.get(0);
                        while(sum < topicRand)
                        {
                            sum+= topicSample.get(++topic);
                        }

                        //sample and index from the topic
                        Vec basisVec = basis.get(topic);
                        int randBasisWord = rand.nextInt(basisVec.nnz());

                        int pos = 0;
                        for(IndexValue iv : basisVec)
                        {
                            if(pos == randBasisWord)
                            {
                                doc.increment(iv.getIndex() , 1.0);
                                break;
                            }
                            pos++;
                        }
                    }

                    docs.add(new DataPoint(doc, new int[0], new CategoricalData[0]));
                }

                //

                OnlineLDAsvi lda = new OnlineLDAsvi();
                lda.setAlpha(0.1);
                lda.setEta(1.0/basis.size());
                lda.setKappa(0.6);
                lda.setMiniBatchSize(256);
                lda.setTau0(64);
                lda.setEpochs(1);


                if(iters == 0)
                    lda.model(new SimpleDataSet(docs), basis.size());
                else
                    lda.model(new SimpleDataSet(docs), basis.size(), ex);

                if(passTest(lda, basis, dirichlet, rand))
                   break; //you did it , skip out of here
            }
            while(attempts-- > 0);
            assertTrue(attempts > 0);
        }
        
        ex.shutdown();
    }

    public boolean passTest(OnlineLDAsvi lda, List<Vec> basis, Dirichlet dirichlet, Random rand)
    {
        //map from the LDA topics to the basis topics
        Map<Integer, Integer> ldaTopicToBasis = new HashMap<Integer, Integer>();
        for(int i = 0; i < lda.getK(); i++)
        {
            Vec topic = lda.getTopicVec(i);
            int minIndx = 0;
            double minDist = topic.subtract(basis.get(0)).pNorm(2);
            for(int j = 1; j < basis.size(); j++)
            {
                double dist = topic.subtract(basis.get(j)).pNorm(2);
                if(dist <minDist)
                {
                    minDist = dist;
                    minIndx = j;
                }
            }
            ldaTopicToBasis.put(i, minIndx);
            if(minDist > 0.025)//values of around 0.1 are when failure happens
                return false;
        }
        
        //make sure no basis was closest to 2 or more topics
        if(basis.size() != new IntSet(ldaTopicToBasis.values()).size())
            return false;
        
        //make sure that computing the topic distirbution works
        for(Vec topicSample : dirichlet.sample(100, rand))
        {
            Vec doc = new DenseVector(basis.get(0).length());
            //sample 40 times
            for(int i = 0; i < 100; i++)
            {
                double topicRand = rand.nextDouble();
                int topic = 0;
                double sum = topicSample.get(0);
                while(sum < topicRand)
                {
                    sum+= topicSample.get(++topic);
                }
                
                //sample and index from the topic
                Vec basisVec = basis.get(topic);
                int randBasisWord = rand.nextInt(basisVec.nnz());
                
                int pos = 0;
                for(IndexValue iv : basisVec)
                {
                    if(pos == randBasisWord)
                    {
                        doc.increment(iv.getIndex(), 1.0);
                        break;
                    }
                    pos++;
                }
            }
            
            Vec ldaTopics = lda.getTopics(doc);
            for(int i = 0; i < ldaTopics.length(); i++)
            {
                double ldaVal = ldaTopics.get(i);
                if(ldaVal > 0.2)
                {
                    if(Math.abs(topicSample.get(ldaTopicToBasis.get(i)) - ldaVal) >  0.25)
                        return false;
                }
            }
        }
        
        return true;
    }

}
