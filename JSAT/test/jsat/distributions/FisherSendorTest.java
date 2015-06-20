/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package jsat.distributions;

import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 *
 * @author Edward Raff
 */
public class FisherSendorTest
{
    double[] range = new double[]
    {
        -3., -2.75, -2.5, -2.25, -2., -1.75, 
        -1.5, -1.25, -1., -0.75, -0.5, -0.25,
        0., 0.25, 0.5, 0.75, 1., 1.25, 1.5, 
        1.75, 2., 2.25, 2.5, 2.75, 3.
    };
    
    public FisherSendorTest()
    {
    }

    @BeforeClass
    public static void setUpClass() throws Exception
    {
    }

    @AfterClass
    public static void tearDownClass() throws Exception
    {
    }
    
    @Before
    public void setUp()
    {
    }

    /**
     * Test of logPdf method, of class FisherSendor.
     */
    @Test
    public void testLogPdf()
    {
        System.out.println("logPdf");
        ContinuousDistribution instance = null;
        
        double[] parmTwo0 = new double[]{0,0,0,0,0,0,0,0,0,0,0,0,0,-1.0755311112886419,-1.6865522751055781,-2.0677264461003304,-2.3502536967514276,-2.5765028780652766,-2.765924303489656,-2.9291924032622623,-3.072846636225469,-3.2012052668045246,-3.3172796396247555,-3.4232587102214747,-3.5207865035324826};
        double[] paramTwo1 = new double[]{0,0,0,0,0,0,0,0,0,0,0,0,0,-0.7312989949608601,-1.3197956283990615,-1.689940033450167,-1.9693449650880337,-2.1981124382434136,-2.394181321271322,-2.5671765210819997,-2.7228852876009073,-2.865072967010014,-2.9963360392653255,-3.1185468636467606,-3.233104431080702};
        double[] paramTwo2 = new double[]{0,0,0,0,0,0,0,0,0,0,0,0,0,-0.8606581655809521,-1.3365909264810165,-1.69115240188437,-1.9693449650880337,-2.197546214702604,-2.390793562926733,-2.5583016221696315,-2.706089989518952,-2.838297232629496,-2.957887102223775,-3.067053290028565,-3.1674642734484513};
        double[] paramTwo3 = new double[]{0,0,0,0,0,0,0,0,0,0,0,0,0,-0.42786617867213905,-0.62825725877403,-0.8879767442017231,-1.1447298858494002,-1.3865072171614459,-1.6114279857379477,-1.8202831852372927,-2.0145516198939206,-2.1957882250863383,-2.3654318837185913,-2.5247554082772825,-2.6748652831951816};
        
        instance = new FisherSendor(0.5, 0.5);
        for(int i = 0; i < range.length; i++)
            assertEquals(parmTwo0[i], instance.logPdf(range[i]), 1e-10);
        instance = new FisherSendor(0.5, 3);
        for(int i = 0; i < range.length; i++)
            assertEquals(paramTwo1[i], instance.logPdf(range[i]), 1e-10);
        instance = new FisherSendor(3, 0.5);
        for(int i = 0; i < range.length; i++)
            assertEquals(paramTwo2[i], instance.logPdf(range[i]), 1e-10);
        instance = new FisherSendor(3, 3);
        for(int i = 0; i < range.length; i++)
            assertEquals(paramTwo3[i], instance.logPdf(range[i]), 1e-10);
    }

    /**
     * Test of pdf method, of class FisherSendor.
     */
    @Test
    public void testPdf()
    {
        System.out.println("pdf");
        ContinuousDistribution instance = null;
        
        double[] parmTwo0 = new double[]{0,0,0,0,0,0,0,0,0,0,0,0,0,0.34111653633834343,0.18515679448477712,0.12647299825855507,0.09534497043772663,0.0760394588478015,0.06291791721274038,0.05344017883680418,0.04628919862119428,0.040713104242116736,0.03625131437838356,0.03260600809421878,0.029576164260098146};
        double[] paramTwo1 = new double[]{0,0,0,0,0,0,0,0,0,0,0,0,0,0.48128339914819057,0.26718990241438084,0.1845305893239915,0.13954823524954643,0.11101250368051968,0.09124735061736179,0.07675194728217233,0.06568496075058096,0.05697897339502477,0.049969820825925305,0.04422138145022476,0.039434885680130864};
        double[] paramTwo2 = new double[]{0,0,0,0,0,0,0,0,0,0,0,0,0,0.4228836632329831,0.2627398430023239,0.18430700582252596,0.13954823524954643,0.11107537937259643,0.09155699880129668,0.07743614465606578,0.06679747560359521,0.058525236030376834,0.051928520997649746,0.04655814627519772,0.042110242695782625};
        double[] paramTwo3 = new double[]{0,0,0,0,0,0,0,0,0,0,0,0,0,0.6518986469044031,0.5335207799449517,0.4114874554751566,0.3183098861837907,0.2499467916526712,0.19960238111582101,0.1619798741292785,0.13338019498623793,0.11127082047753811,0.09390873395589414,0.08007789678230687,0.068916111927724};
        
        instance = new FisherSendor(0.5, 0.5);
        for(int i = 0; i < range.length; i++)
            assertEquals(parmTwo0[i], instance.pdf(range[i]), 1e-10);
        instance = new FisherSendor(0.5, 3);
        for(int i = 0; i < range.length; i++)
            assertEquals(paramTwo1[i], instance.pdf(range[i]), 1e-10);
        instance = new FisherSendor(3, 0.5);
        for(int i = 0; i < range.length; i++)
            assertEquals(paramTwo2[i], instance.pdf(range[i]), 1e-10);
        instance = new FisherSendor(3, 3);
        for(int i = 0; i < range.length; i++)
            assertEquals(paramTwo3[i], instance.pdf(range[i]), 1e-10);
    }

    /**
     * Test of cdf method, of class FisherSendor.
     */
    @Test
    public void testCdf()
    {
        System.out.println("cdf");
        ContinuousDistribution instance = null;
        
        double[] parmTwo0 = new double[]{0,0,0,0,0,0,0,0,0,0,0,0,0,0.3727156131603777,0.4345598848424255,0.4726180836933109,0.49999999999999967,0.5212535928736133,0.5385275931078592,0.5530131159980487,0.5654401151575745,0.5762881886500197,0.5858890925015626,0.594481626266527,0.6022432216826443};
        double[] paramTwo1 = new double[]{0,0,0,0,0,0,0,0,0,0,0,0,0,0.5096165742200301,0.5978642569008251,0.6531118031726221,0.6931449079784125,0.7242266764500165,0.7493719694170181,0.770285305382158,0.7880316369300455,0.8033234720594439,0.8166619182600992,0.8284130726388325,0.8388525357634297};
        double[] paramTwo2 = new double[]{0,0,0,0,0,0,0,0,0,0,0,0,0,0.12886711217619012,0.21196836306995448,0.26683409112435963,0.30685509202158756,0.3379412545043875,0.36312957646202815,0.38416534040879513,0.4021357430991749,0.41776015689702894,0.4315373428052034,0.443826221767879,0.45489304152075205};
        double[] paramTwo3 = new double[]{0,0,0,0,0,0,0,0,0,0,0,0,0,0.14237848993264704,0.2917914057909287,0.40936461121441453,0.49999999999999983,0.5705897131786168,0.6264699609476689,0.6714465399972,0.7082085942090712,0.7386753434255021,0.764238177345735,0.7859230176565496,0.8044988905221148};
        
        instance = new FisherSendor(0.5, 0.5);
        for(int i = 0; i < range.length; i++)
            assertEquals(parmTwo0[i], instance.cdf(range[i]), 1e-10);
        instance = new FisherSendor(0.5, 3);
        for(int i = 0; i < range.length; i++)
            assertEquals(paramTwo1[i], instance.cdf(range[i]), 1e-10);
        instance = new FisherSendor(3, 0.5);
        for(int i = 0; i < range.length; i++)
            assertEquals(paramTwo2[i], instance.cdf(range[i]), 1e-10);
        instance = new FisherSendor(3, 3);
        for(int i = 0; i < range.length; i++)
            assertEquals(paramTwo3[i], instance.cdf(range[i]), 1e-10);
    }

    /**
     * Test of invCdf method, of class FisherSendor.
     */
    @Test
    public void testInvCdf()
    {
        System.out.println("invCdf");
        ContinuousDistribution instance = null;
        
        double[] parmTwo0 = new double[]{5.334203456719422e-8,0.00006913318696422088,0.0007812247277505513,0.003500715803229859,0.010417164444029625,0.024615063171980367,0.0502444629207317,0.09287319888065815,0.16014839504292144,0.2629821687138465,0.41764853333136576,0.6495302481781395,1.000000000000003,1.5395741811946149,2.394357743875021,3.802539179331621,6.244208689896571,10.767368972452411,19.90269060249784,40.62553051410859,95.9954127030347,285.65586474553135,1280.041407392871,/*last 2 removed for being too large, relative accuray is still good 14464.832939334676,1.874694148340569e7*/};
        double[] paramTwo1 = new double[]{1.5805047937078598e-8,0.00002048343872385061,0.00023141418709826667,0.0010360499598851547,0.003075987604420405,0.007234721118664744,0.014646188270129645,0.026711949496745824,0.04513199564708614,0.07195612582172162,0.10966587338583667,0.16130199781329901,0.23066126571683299,0.3226018002981843,0.4435251089394354,0.6021587290154584,0.8108775237950727,1.088050726896899,1.4624918050988818,1.9826371274899546,2.7377289774408404,3.91504731419958,5.996708740298692,10.807066928600696,44.75298856639894};
        double[] paramTwo2 = new double[]{0.02234487644364417,0.09253204468952805,0.16675814072473877,0.25542475473363396,0.3652662510570255,0.5043787318085855,0.6837645151333946,0.9190747961282854,1.2332318638205622,1.6606916944225336,2.2546637830521425,3.0997967124662353,4.335361626028854,6.199551236541197,9.11860699346005,13.897357432466515,22.15722982470372,37.436428970554886,68.27715044736316,138.22232862855972,325.0988393330919, /*same story, see param0 965.2044193999906,4321.256239909707,48819.92782127823,6.327092862316094e7 */};
        double[] paramTwo3 = new double[]{0.029582973445028582,0.10646321922955648,0.17053427217757644,0.23282567658262066,0.29615129152974795,0.3620484899223977,0.4316934431570105,0.5061867593792821,0.5866920309095274,0.674534975988236,0.7713012301239432,0.8789549312280995,1.,1.1377147615552636,1.2965103139266438,1.4825028139347958,1.7044717625527248,1.9755554278548542,2.31645862556289,2.7620609609898974,3.376652503639516,4.295058924246869,5.863923932889599,9.39291529259317,33.80322812573971};
        
        instance = new FisherSendor(0.5, 0.5);
        for(int i = 0; i < parmTwo0.length; i++)
            assertEquals(parmTwo0[i], instance.invCdf(range[i]/6.1+0.5), 1e-8);
        instance = new FisherSendor(0.5, 3);
        for(int i = 0; i < range.length; i++)
            assertEquals(paramTwo1[i], instance.invCdf(range[i]/6.1+0.5), 1e-10);
        instance = new FisherSendor(3, 0.5);
        for(int i = 0; i < paramTwo2.length; i++)
            assertEquals(paramTwo2[i], instance.invCdf(range[i]/6.1+0.5), 1e-10);
        instance = new FisherSendor(3, 3);
        for(int i = 0; i < range.length; i++)
            assertEquals(paramTwo3[i], instance.invCdf(range[i]/6.1+0.5), 1e-10);
    }

    /**
     * Test of min method, of class FisherSendor.
     */
    @Test
    public void testMin()
    {
        System.out.println("min");
        ContinuousDistribution dist = new FisherSendor(0.5, 3);
        assertTrue(0 == dist.min());
    }

    /**
     * Test of max method, of class FisherSendor.
     */
    @Test
    public void testMax()
    {
        System.out.println("max");
        ContinuousDistribution dist = new FisherSendor(0.5, 3);
        assertTrue(Double.POSITIVE_INFINITY == dist.max());
    }

    /**
     * Test of mean method, of class FisherSendor.
     */
    @Test
    public void testMean()
    {
        System.out.println("mean");
        ContinuousDistribution dist = new FisherSendor(0.5, 0.5);
        assertEquals(Double.NaN, dist.mean(), 1e-10);
        dist = new FisherSendor(0.5, 3);
        assertEquals(3, dist.mean(), 1e-10);
        dist = new FisherSendor(3, 0.5);
        assertEquals(Double.NaN, dist.mean(), 1e-10);
        dist = new FisherSendor(3, 3);
        assertEquals(3, dist.mean(), 1e-10);
    }

    /**
     * Test of median method, of class FisherSendor.
     */
    @Test
    public void testMedian()
    {
        System.out.println("median");
        ContinuousDistribution dist = new FisherSendor(0.5, 0.5);
        assertEquals(1.000000000000003, dist.median(), 1e-10);
        dist = new FisherSendor(0.5, 3);
        assertEquals(0.23066126571683299, dist.median(), 1e-10);
        dist = new FisherSendor(3, 0.5);
        assertEquals(4.335361626028854, dist.median(), 1e-10);
        dist = new FisherSendor(3, 3);
        assertEquals(1, dist.median(), 1e-10);
    }

    /**
     * Test of mode method, of class FisherSendor.
     */
    @Test
    public void testMode()
    {
        System.out.println("mode");
        ContinuousDistribution dist = new FisherSendor(0.5, 0.5);
        assertEquals(Double.NaN, dist.mode(), 1e-10);
        dist = new FisherSendor(0.5, 3);
        assertEquals(Double.NaN, dist.mode(), 1e-10);
        dist = new FisherSendor(3, 0.5);
        assertEquals(2.0/3.0/10, dist.mode(), 1e-10);
        dist = new FisherSendor(3, 3);
        assertEquals(0.2, dist.mode(), 1e-10);
    }

    /**
     * Test of variance method, of class FisherSendor.
     */
    @Test
    public void testVariance()
    {
        System.out.println("variance");
        ContinuousDistribution dist = new FisherSendor(0.5, 0.5);
        assertEquals(Double.NaN, dist.variance(), 1e-10);
        dist = new FisherSendor(0.5, 3);
        assertEquals(Double.NaN, dist.variance(), 1e-10);
        dist = new FisherSendor(3, 0.5);
        assertEquals(Double.NaN, dist.variance(), 1e-10);
        dist = new FisherSendor(3, 3);
        assertEquals(Double.NaN, dist.variance(), 1e-10);
        dist = new FisherSendor(6, 4);
        assertEquals(Double.NaN, dist.variance(), 1e-10);
        dist = new FisherSendor(4, 6);
        assertEquals(9.0/2.0, dist.variance(), 1e-10);
    }

    /**
     * Test of skewness method, of class FisherSendor.
     */
    @Test
    public void testSkewness()
    {
        System.out.println("skewness");
        ContinuousDistribution dist = new FisherSendor(0.5, 0.5);
        assertEquals(Double.NaN, dist.skewness(), 1e-10);
        dist = new FisherSendor(0.5, 3);
        assertEquals(Double.NaN, dist.skewness(), 1e-10);
        dist = new FisherSendor(3, 0.5);
        assertEquals(Double.NaN, dist.skewness(), 1e-10);
        dist = new FisherSendor(3, 3);
        assertEquals(Double.NaN, dist.skewness(), 1e-10);
        dist = new FisherSendor(6, 6);
        assertEquals(Double.NaN, dist.skewness(), 1e-10);
        dist = new FisherSendor(5, 7);
        assertEquals(10.39230484541326, dist.skewness(), 1e-10);
        dist = new FisherSendor(7, 5);
        assertEquals(Double.NaN, dist.skewness(), 1e-10);
    }
    @Test
    public void testEquals(){
    	System.out.println("equals");
    	ContinuousDistribution d1 = new FisherSendor(0.5, 0.5);
    	ContinuousDistribution d2 = new FisherSendor(0.6, 0.5);
    	ContinuousDistribution d3 = new FisherSendor(0.5, 0.6);
    	ContinuousDistribution d4 = new FisherSendor(0.5, 0.5);
    	Integer i = new Integer(1);
    	assertFalse(d1.equals(d2));
    	assertFalse(d1.equals(d3));
    	assertFalse(d2.equals(d3));
    	assertFalse(d1.equals(i));
    	assertFalse(d1.equals(null));
    	assertEquals(d1, d1);
    	assertEquals(d1, d4);
    	assertEquals(d1, d1.clone());
    }
    
    @Test
    public void testHashCode(){
    	System.out.println("hashCode");
    	ContinuousDistribution d1 = new FisherSendor(0.5, 0.5);
    	ContinuousDistribution d2 = new FisherSendor(0.6, 0.5);
    	ContinuousDistribution d4 = new FisherSendor(0.5, 0.5);
    	assertEquals(d1.hashCode(), d4.hashCode());
    	assertFalse(d1.hashCode()==d2.hashCode());
    }
}
