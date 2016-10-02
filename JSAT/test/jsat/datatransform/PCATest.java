/*
 * Copyright (C) 2016 Edward Raff <Raff.Edward@gmail.com>
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
package jsat.datatransform;

import java.util.Random;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.ClassificationModelEvaluation;
import jsat.classifiers.Classifier;
import jsat.classifiers.knn.NearestNeighbour;
import jsat.distributions.Normal;
import jsat.utils.GridDataGenerator;
import jsat.utils.random.XORWOW;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 *
 * @author Edward Raff <Raff.Edward@gmail.com>
 */
public class PCATest
{
    
    public PCATest()
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
     * Test of transform method, of class PCA.
     */
    @Test
    public void testTransform()
    {
        System.out.println("transform");
        GridDataGenerator gdg = new GridDataGenerator(new Normal(0, 0.05), new Random(12), 1, 1, 1);
        ClassificationDataSet easyTrain = new ClassificationDataSet(gdg.generateData(80).getBackingList(), 0);
        ClassificationDataSet easyTest = new ClassificationDataSet(gdg.generateData(10).getBackingList(), 0);
        
        //lets project the data into a higher dimension
        JLTransform jl = new  JLTransform(30, JLTransform.TransformMode.GAUSS);
        jl.fit(easyTrain);
        easyTrain.applyTransform(jl);
        easyTest.applyTransform(jl);
        
        ClassificationModelEvaluation cme = new ClassificationModelEvaluation(new NearestNeighbour(3), easyTrain);
        cme.evaluateTestSet(easyTest);
        double errorRate = cme.getErrorRate();
        
        PCA pca = new PCA(10);
        pca.fit(easyTrain);
        assertEquals(10, pca.transform(easyTrain.getDataPoint(0)).getNumericalValues().length());
        
        
        cme = new ClassificationModelEvaluation(new DataModelPipeline((Classifier)new NearestNeighbour(3), new PCA(10)), easyTrain);
        cme.evaluateTestSet(easyTest);
        assertTrue(cme.getErrorRate() < (errorRate+0.01)*1.05);
        
        cme = new ClassificationModelEvaluation(new DataModelPipeline((Classifier)new NearestNeighbour(3), new PCA(3)), easyTrain);
        cme.evaluateTestSet(easyTest);
        assertTrue(cme.getErrorRate() < (errorRate+0.01)*1.05);
    }

    
}
