/*
 * Copyright (C) 2018 Edward Raff
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
package jsat.distributions;

import jsat.linear.Vec;
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
public class KumaraswamyTest
{
    double[] range = new double[]
    {
        0, 0.050000000000000000000, 0.10000000000000000000, 0.15000000000000000000, 0.20000000000000000000, 
        0.25000000000000000000, 0.30000000000000000000, 0.35000000000000000000, 0.40000000000000000000, 
        0.45000000000000000000, 0.50000000000000000000, 0.55000000000000000000, 0.60000000000000000000, 
        0.65000000000000000000, 0.70000000000000000000, 0.75000000000000000000, 0.80000000000000000000, 
        0.85000000000000000000, 0.90000000000000000000, 0.95000000000000000000, 1.0000000000000000000
    };

    public KumaraswamyTest()
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
     * Test of pdf method, of class Beta.
     */
    @Test
    public void testPdf()
    {
        System.out.println("pdf");
        ContinuousDistribution instance = null;
        
        double[] parmTwo0 = new double[]{0,1.2688612786305156476,0.95605808387038662777,0.82465035211625394177,0.75187619375943209207,0.70710678118654752440,0.67869854393773691109,0.66125255318686165196,0.65201121779749292660,0.64955703500907604881,0.65328148243818826393,0.66317644115150481346,0.67980478762776913537,0.70442591676201416669,0.73934092805487665228,0.78867513459481288225,0.86023870029448346138,0.97063614821427270704,1.1632937248661048310,1.6119126569980934130,0};
        double[] paramTwo1 = new double[]{0,4.0436141291243375437,2.2177581392778258978,1.4539308481385294180,1.0249223594996214535,0.75000000000000000000,0.56019662378357973747,0.42287473165049213891,0.32039154317679829860,0.24229856737469505979,0.18198051533946385980,0.13502936045407570105,0.098386676965933508143,0.069859681082909290709,0.047832953802703781920,0.031088913245535263673,0.018691769624716090152,0.0099101022338543865007,0.0041637771599603653989,0.00098667984907572921077,0};
        double[] paramTwo2 = new double[]{0,0.0037502343969749453187,0.015007505629691605257,0.033807097694255976552,0.060241449667687414717,0.094491118252306806804,0.13686025610801962669,0.18782051096329649332,0.24806946917841691238,0.31861352523454621291,0.40089186286863657703,0.49697125857422871843,0.60986783446104458546,0.74410921107585204676,0.90678568249206644591,1.1096931643111617000,1.3742360023532837230,1.7446398544038663636,2.3339504543257195014,3.5845995086146493566,0};
        double[] paramTwo3 = new double[]{0,0.022494375351562500000,0.089820090000000000000,0.20113543160156250000,0.35426304000000000000,0.54505920410156250000,0.76685049000000000000,1.0099873128515625000,1.2615782400000000000,1.5054829878515625000,1.7226562500000000000,1.8919486691015625000,1.9914854400000000000,2.0007572066015625000,1.9035720900000000000,1.6920318603515625000,1.3717094400000000000,0.96821910035156250000,0.53538489000000000000,0.16522700660156250000,0};
        
        instance = new Kumaraswamy(0.5, 0.5);
        for(int i = 0; i < range.length; i++)
            assertEquals(parmTwo0[i], instance.pdf(range[i]), 1e-10);
        instance = new Kumaraswamy(0.5, 3);
        for(int i = 0; i < range.length; i++)
            assertEquals(paramTwo1[i], instance.pdf(range[i]), 1e-10);
        instance = new Kumaraswamy(3, 0.5);
        for(int i = 0; i < range.length; i++)
            assertEquals(paramTwo2[i], instance.pdf(range[i]), 1e-10);
        instance = new Kumaraswamy(3, 3);
        for(int i = 0; i < range.length; i++)
            assertEquals(paramTwo3[i], instance.pdf(range[i]), 1e-10);
    }

    /**
     * Test of cdf method, of class Beta.
     */
    @Test
    public void testCdf()
    {
        System.out.println("cdf");
        ContinuousDistribution instance = null;
        
        double[] parmTwo0 = new double[]{0,0.11886822651204943036,0.17309478536947049508,0.21724737919361885315,0.25650393107963101974,0.29289321881345247560,0.32748424368284583176,0.36094442988888799159,0.39374554189983548328,0.42625824036413116334,0.45880389985380301560,0.49168892271520020992,0.52523339338311016489,0.55980206137449367338,0.59584659661717006392,0.63397459621556135324,0.67508030376709367384,0.72063365580172104597,0.77346809948820408927,0.84087562877075174656,1.0000000000000000000};
        double[] paramTwo1 = new double[]{0,0.53200073313743585740,0.68030607465219759292,0.76998975405533631883,0.83108350559986540570,0.87500000000000000000,0.90748443976704817441,0.93188672733837137426,0.95034880891449794576,0.96433035671228233578,0.97487373415291633540,0.98275046291896034679,0.98854800926934015733,0.99272407812897062311,0.99564209817607952752,0.99759526419164492536,0.99882332579968033854,0.99952461605776161435,0.99986486239700381844,0.99998376619954074320,1.0000000000000000000};
        double[] paramTwo2 = new double[]{0,0.000062501953247079850078,0.00050012506253908986427,0.0016889262359151813113,0.0040080321609014100131,0.0078432583507785285619,0.013592376347384468371,0.021672345274856292944,0.032529070204174041716,0.046650641160335179407,0.064585653306514653604,0.086969332388007899944,0.11456225515285378704,0.14831050258911845614,0.18944463483362223815,0.23965468371272253888,0.30143003213708077415,0.37881162277454031553,0.47942339660718519764,0.62234274798436612042,1.0000000000000000000};
        double[] paramTwo3 = new double[]{0,0.00037495312695312500000,0.0029970010000000000000,0.010090866568359375000,0.023808512000000000000,0.046146392822265625000,0.078832683000000000000,0.12318901876367187500,0.17997414400000000000,0.24922038376757812500,0.33007812500000000000,0.42068844470898437500,0.51810969600000000000,0.61833024096289062500,0.71640660700000000000,0.80677413940429687500,0.88378572800000000000,0.94254339940820312500,0.98009748900000000000,0.99709873784960937500,1.0000000000000000000};
        
        instance = new Kumaraswamy(0.5, 0.5);
        for(int i = 0; i < range.length; i++)
        {
            assertEquals(parmTwo0[i], instance.cdf(range[i]), 1e-10);
            assertEquals(range[i], instance.invCdf(parmTwo0[i]), 1e-10);
        }
        
        instance = new Kumaraswamy(0.5, 3);
        for(int i = 0; i < range.length; i++)
        {
            assertEquals(paramTwo1[i], instance.cdf(range[i]), 1e-10);
            assertEquals(range[i], instance.invCdf(paramTwo1[i]), 1e-10);
        }
        instance = new Kumaraswamy(3, 0.5);
        for(int i = 0; i < range.length; i++)
        {
            assertEquals(paramTwo2[i], instance.cdf(range[i]), 1e-10);
            assertEquals(range[i], instance.invCdf(paramTwo2[i]), 1e-10);
        }
        instance = new Kumaraswamy(3, 3);
        for(int i = 0; i < range.length; i++)
        {
            assertEquals(paramTwo3[i], instance.cdf(range[i]), 1e-10);
            assertEquals(range[i], instance.invCdf(paramTwo3[i]), 1e-10);
        }
    }


    /**
     * Test of min method, of class Beta.
     */
    @Test
    public void testMin()
    {
        System.out.println("min");
        ContinuousDistribution dist = new Kumaraswamy(0.5, 3);
        assertTrue(0 == dist.min());
    }

    /**
     * Test of max method, of class Beta.
     */
    @Test
    public void testMax()
    {
        System.out.println("max");
        ContinuousDistribution dist = new Kumaraswamy(0.5, 3);
        assertTrue(1 == dist.max());
    }

    /**
     * Test of mean method, of class Beta.
     */
    @Test
    public void testMean()
    {
        System.out.println("mean");
        ContinuousDistribution dist = new Kumaraswamy(0.5, 0.5);
        assertEquals(0.53333333333333333333, dist.mean(), 1e-10);
        dist = new Kumaraswamy(0.5, 3);
        assertEquals(0.10000000000000000000, dist.mean(), 1e-10);
        dist = new Kumaraswamy(3, 0.5);
        assertEquals(0.84130926319527255671, dist.mean(), 1e-10);
        dist = new Kumaraswamy(3, 3);
        assertEquals(0.57857142857142857143, dist.mean(), 1e-10);
    }

    /**
     * Test of median method, of class Beta.
     */
    @Test
    public void testMedian()
    {
        System.out.println("median");
        ContinuousDistribution dist = new Kumaraswamy(0.5, 0.5);
        assertEquals(0.56250000000000000000, dist.median(), 1e-10);
        dist = new Kumaraswamy(0.5, 3);
        assertEquals(0.042559472979237107632, dist.median(), 1e-10);
        dist = new Kumaraswamy(3, 0.5);
        assertEquals(0.90856029641606982945, dist.median(), 1e-10);
        dist = new Kumaraswamy(3, 3);
        assertEquals(0.59088011327517717570, dist.median(), 1e-10);
    }

    /**
     * Test of mode method, of class Beta.
     */
    @Test
    public void testMode()
    {
        System.out.println("mode");
        ContinuousDistribution dist = new Kumaraswamy(0.5, 0.5);
        assertTrue(Double.isNaN(dist.mode()));
        dist = new Kumaraswamy(0.5, 3);
        assertTrue(Double.isNaN(dist.mode()));
        dist = new Kumaraswamy(3, 0.5);
        assertTrue(Double.isNaN(dist.mode()));
        dist = new Kumaraswamy(3, 3);
        assertEquals(0.62996052494743658238, dist.mode(), 1e-10);
    }

    /**
     * Test of variance method, of class Beta.
     */
    @Test
    public void testVariance()
    {
        System.out.println("variance");
        ContinuousDistribution dist = new Kumaraswamy(0.5, 0.5);
        assertEquals(0.12190476190476190476, dist.variance(), 1e-10);
        dist = new Kumaraswamy(0.5, 3);
        assertEquals(0.018571428571428571429, dist.variance(), 1e-10);
        dist = new Kumaraswamy(3, 0.5);
        assertEquals(0.031372883441767762274, dist.variance(), 1e-10);
        dist = new Kumaraswamy(3, 3);
        assertEquals(0.033436920222634508349, dist.variance(), 1e-10);
    }

    /**
     * Test of skewness method, of class Beta.
     */
    @Test
    public void testSkewness()
    {
        System.out.println("skewness");
        
        double tol = 1e-10;
        
        ContinuousDistribution dist = new Kumaraswamy(0.5, 0.5);
        assertEquals(-0.13530526527453140657, dist.skewness(), tol);
        dist = new Kumaraswamy(0.5, 3);
        assertEquals(2.1073213127948306253, dist.skewness(), tol);
        dist = new Kumaraswamy(3, 0.5);
        assertEquals(-1.4389341273489145202, dist.skewness(), tol);
        dist = new Kumaraswamy(3, 3);
        assertEquals(-0.27980354819056016487, dist.skewness(), tol);
    }
    
}
