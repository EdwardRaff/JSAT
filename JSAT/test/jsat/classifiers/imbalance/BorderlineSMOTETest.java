/*
 * Copyright (C) 2017 Edward Raff
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package jsat.classifiers.imbalance;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import jsat.FixedProblems;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.DataPointPair;
import jsat.classifiers.linear.LogisticRegressionDCD;
import jsat.utils.SystemInfo;
import jsat.utils.random.RandomUtil;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 * To make sure these tests exercise the danger list code in SMOTE, we push the
 * two target classes closer together during training time to increase overlap.
 * But then use a wider separate at testing time to make sure we get a
 * consistently runnable test case.
 *
 * @author Edward Raff
 */
public class BorderlineSMOTETest
{
    static boolean parallelTrain = true;
    public BorderlineSMOTETest()
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

    
    @Test
    public void testTrainC_ClassificationDataSet_ExecutorService()
    {
        System.out.println("trainC");
        ClassificationDataSet train = FixedProblems.get2ClassLinear2D(200, 20, 0.5, RandomUtil.getRandom());
        
        BorderlineSMOTE smote = new BorderlineSMOTE(new LogisticRegressionDCD(), false);
        smote.train(train, parallelTrain);
        
        
        ClassificationDataSet test = FixedProblems.get2ClassLinear2D(200, 200, 4.0, RandomUtil.getRandom());
        
        for(DataPointPair<Integer> dpp : test.getAsDPPList())
            assertEquals(dpp.getPair().longValue(), smote.classify(dpp.getDataPoint()).mostLikely());
        
        smote = smote.clone();
        
        for(DataPointPair<Integer> dpp : test.getAsDPPList())
            assertEquals(dpp.getPair().longValue(), smote.classify(dpp.getDataPoint()).mostLikely());
        
        smote = new BorderlineSMOTE(new LogisticRegressionDCD(), true);
        smote.train(train, parallelTrain);
        
        for(DataPointPair<Integer> dpp : test.getAsDPPList())
            assertEquals(dpp.getPair().longValue(), smote.classify(dpp.getDataPoint()).mostLikely());
        
        smote = smote.clone();
        
        for(DataPointPair<Integer> dpp : test.getAsDPPList())
            assertEquals(dpp.getPair().longValue(), smote.classify(dpp.getDataPoint()).mostLikely());
    }
    
    @Test
    public void testTrainC_ClassificationDataSet()
    {
        System.out.println("trainC");
        ClassificationDataSet train = FixedProblems.get2ClassLinear2D(200, 20, 0.5, RandomUtil.getRandom());
        
        BorderlineSMOTE smote = new BorderlineSMOTE(new LogisticRegressionDCD(), true);
        smote.train(train);
        
        ClassificationDataSet test = FixedProblems.get2ClassLinear2D(200, 200, 4.0, RandomUtil.getRandom());
        
        for(DataPointPair<Integer> dpp : test.getAsDPPList())
            assertEquals(dpp.getPair().longValue(), smote.classify(dpp.getDataPoint()).mostLikely());
        
        smote = smote.clone();
        
        for(DataPointPair<Integer> dpp : test.getAsDPPList())
            assertEquals(dpp.getPair().longValue(), smote.classify(dpp.getDataPoint()).mostLikely());
        
        smote = new BorderlineSMOTE(new LogisticRegressionDCD(), false);
        smote.train(train);
        
        for(DataPointPair<Integer> dpp : test.getAsDPPList())
            assertEquals(dpp.getPair().longValue(), smote.classify(dpp.getDataPoint()).mostLikely());
        
        smote = smote.clone();
        
        for(DataPointPair<Integer> dpp : test.getAsDPPList())
            assertEquals(dpp.getPair().longValue(), smote.classify(dpp.getDataPoint()).mostLikely());
    }
}
