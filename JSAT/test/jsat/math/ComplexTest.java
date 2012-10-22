/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package jsat.math;

import org.junit.After;
import org.junit.AfterClass;
import static org.junit.Assert.*;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

/**
 *
 * @author Edward Raff
 */
public class ComplexTest
{
    Complex a, b, c, d, aClone, bClone, cClone, dClone;
    
    public ComplexTest()
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
        a = new Complex(2, 1);
        aClone = a.clone();
        b = new Complex(3, 7);
        bClone = b.clone();
        c = new Complex(9, -4);
        cClone = c.clone();
        d = new Complex(-5, 3);
        dClone = d.clone();
    }
    
    @After
    public void tearDown()
    {
    }

    /**
     * Test of I method, of class Complex.
     */
    @Test
    public void testI()
    {
        System.out.println("I");
        Complex result = Complex.I();
        assertEquals(0.0, result.getReal(), 0.0);
        assertEquals(1.0, result.getImag(), 0.0);
    }

    /**
     * Test of getReal method, of class Complex.
     */
    @Test
    public void testGetReal()
    {
        System.out.println("getReal");
        assertEquals(2.0, a.getReal(), 0.0);
        assertEquals(3.0, b.getReal(), 0.0);
        assertEquals(9.0, c.getReal(), 0.0);
        assertEquals(-5.0, d.getReal(), 0.0);
    }

    /**
     * Test of getImag method, of class Complex.
     */
    @Test
    public void testGetImag()
    {
        System.out.println("getImag");
        assertEquals(1.0, a.getImag(), 0.0);
        assertEquals(7.0, b.getImag(), 0.0);
        assertEquals(-4.0, c.getImag(), 0.0);
        assertEquals(3.0, d.getImag(), 0.0);
    }

    /**
     * Test of mutableAdd method, of class Complex.
     */
    @Test
    public void testMutableAdd_double_double()
    {
        System.out.println("mutableAdd");
        a.mutableAdd(b.getReal(), b.getImag());
        assertEquals(new Complex(5, 8), a);
        assertEquals(new Complex(3, 7), b);
        c.mutableAdd(d.getReal(), d.getImag());
        assertEquals(new Complex(4, -1), c);
        assertEquals(new Complex(-5, 3), d);
    }

    /**
     * Test of mutableAdd method, of class Complex.
     */
    @Test
    public void testMutableAdd_Complex()
    {
        System.out.println("mutableAdd");
        a.mutableAdd(b);
        assertEquals(new Complex(5, 8), a);
        assertEquals(new Complex(3, 7), b);
        c.mutableAdd(d);
        assertEquals(new Complex(4, -1), c);
        assertEquals(new Complex(-5, 3), d);
    }

    /**
     * Test of add method, of class Complex.
     */
    @Test
    public void testAdd()
    {
        System.out.println("add");
        Complex result = a.add(b);
        assertEquals(result, new Complex(5, 8));
        assertEquals(aClone, a);
        assertEquals(bClone, b);
        result = c.add(d);
        assertEquals(result, new Complex(4, -1));
        assertEquals(cClone, c);
        assertEquals(dClone, d);
    }

    /**
     * Test of mutableSubtract method, of class Complex.
     */
    @Test
    public void testMutableSubtract_double_double()
    {
        System.out.println("mutableSubtract");
        a.mutableSubtract(b.getReal(), b.getImag());
        assertEquals(new Complex(-1, -6), a);
        assertEquals(bClone, b);
        c.mutableSubtract(d.getReal(), d.getImag());
        assertEquals(new Complex(14, -7), c);
        assertEquals(dClone, d);
    }

    /**
     * Test of mutableSubtract method, of class Complex.
     */
    @Test
    public void testMutableSubtract_Complex()
    {
        System.out.println("mutableSubtract");
        a.mutableSubtract(b);
        assertEquals(new Complex(-1, -6), a);
        assertEquals(bClone, b);
        c.mutableSubtract(d);
        assertEquals(new Complex(14, -7), c);
        assertEquals(dClone, d);
    }

    /**
     * Test of subtract method, of class Complex.
     */
    @Test
    public void testSubtract()
    {
        System.out.println("subtract");
        Complex result = a.subtract(b);
        assertEquals(result, new Complex(-1, -6));
        assertEquals(aClone, a);
        assertEquals(bClone, b);
        result = c.subtract(d);
        assertEquals(result, new Complex(14, -7));
        assertEquals(cClone, c);
        assertEquals(dClone, d);
    }

    /**
     * Test of cMul method, of class Complex.
     */
    @Test
    public void testCMul()
    {
        System.out.println("cMul");
        double r0 = a.getReal();
        double i0 = a.getImag();
        double r1 = b.getReal();
        double i1 = b.getImag();
        double[] results = new double[2];
        Complex.cMul(r0, i0, r1, i1, results);
        assertEquals(-1.0, results[0], 0.0);
        assertEquals(17.0, results[1], 0.0);
    }

    /**
     * Test of mutableMultiply method, of class Complex.
     */
    @Test
    public void testMutableMultiply_double_double()
    {
        System.out.println("mutableMultiply");
        a.mutableMultiply(b.getReal(), b.getImag());
        assertEquals(new Complex(-1, 17), a);
        assertEquals(bClone, b);
        c.mutableMultiply(d.getReal(), d.getImag());
        assertEquals(new Complex(-33, 47), c);
        assertEquals(dClone, d);
    }

    /**
     * Test of mutableMultiply method, of class Complex.
     */
    @Test
    public void testMutableMultiply_Complex()
    {
        System.out.println("mutableMultiply");
        a.mutableMultiply(b);
        assertEquals(new Complex(-1, 17), a);
        assertEquals(bClone, b);
        c.mutableMultiply(d);
        assertEquals(new Complex(-33, 47), c);
        assertEquals(dClone, d);
    }

    /**
     * Test of multiply method, of class Complex.
     */
    @Test
    public void testMultiply()
    {
        System.out.println("multiply");
        Complex result = a.multiply(b);
        assertEquals(result, new Complex(-1, 17));
        assertEquals(aClone, a);
        assertEquals(bClone, b);
        result = c.multiply(d);
        assertEquals(result, new Complex(-33, 47));
        assertEquals(cClone, c);
        assertEquals(dClone, d);
    }

    /**
     * Test of cDiv method, of class Complex.
     */
    @Test
    public void testCDiv()
    {
        System.out.println("cDiv");
        double r0 = a.getReal();
        double i0 = a.getImag();
        double r1 = b.getReal();
        double i1 = b.getImag();
        double[] results = new double[2];
        Complex.cDiv(r0, i0, r1, i1, results);
        assertEquals( 0.224137931034483, results[0], 1e-14);
        assertEquals(-0.189655172413793, results[1], 1e-14);
    }

    /**
     * Test of mutableDivide method, of class Complex.
     */
    @Test
    public void testMutableDivide_double_double()
    {
        System.out.println("mutableDivide");
        a.mutableDivide(b.getReal(), b.getImag());
        assertTrue(a.equals(new Complex(0.224137931034483, -0.189655172413793), 1e-14));
        assertEquals(bClone, b);
        c.mutableDivide(d.getReal(), d.getImag());
        assertTrue(c.equals(new Complex(-1.67647058823529, -0.205882352941176), 1e-14));
        assertEquals(dClone, d);
    }

    /**
     * Test of mutableDivide method, of class Complex.
     */
    @Test
    public void testMutableDivide_Complex()
    {
        System.out.println("mutableDivide");
        a.mutableDivide(b);
        assertTrue(a.equals(new Complex(0.224137931034483, -0.189655172413793), 1e-14));
        assertEquals(bClone, b);
        c.mutableDivide(d);
        assertTrue(c.equals(new Complex(-1.67647058823529, -0.205882352941176), 1e-14));
        assertEquals(dClone, d);
    }

    /**
     * Test of divide method, of class Complex.
     */
    @Test
    public void testDivide()
    {
        System.out.println("divide");
        Complex result = a.divide(b);
        assertTrue(result.equals(new Complex(0.224137931034483, -0.189655172413793), 1e-14));
        assertEquals(aClone, a);
        assertEquals(bClone, b);
        result = c.divide(d);
        assertTrue(result.equals(new Complex(-1.67647058823529, -0.205882352941176), 1e-14));
        assertEquals(cClone, c);
        assertEquals(dClone, d);
    }

    /**
     * Test of getMagnitude method, of class Complex.
     */
    @Test
    public void testGetMagnitude()
    {
        System.out.println("getMagnitude");
        assertEquals(2.23606797749979, a.getMagnitude(), 1e-14);
        assertEquals(7.61577310586391, b.getMagnitude(), 1e-14);
        assertEquals(9.8488578017961, c.getMagnitude(), 1e-14);
        assertEquals(5.8309518948453, d.getMagnitude(), 1e-14);
    }

    /**
     * Test of getArg method, of class Complex.
     */
    @Test
    public void testGetArg()
    {
        System.out.println("getArg");
        assertEquals(0.463647609000806, a.getArg(), 1e-14);
        assertEquals(1.16590454050981, b.getArg(), 1e-14);
        assertEquals(-0.418224329579229, c.getArg(), 1e-14);
        assertEquals(2.60117315331921, d.getArg(), 1e-14);
    }

    /**
     * Test of mutateConjugate method, of class Complex.
     */
    @Test
    public void testMutateConjugate()
    {
        System.out.println("mutateConjugate");
        a.mutateConjugate();
        assertEquals(new Complex(aClone.getReal(), -aClone.getImag()), a);
        
        c.mutateConjugate();
        assertEquals(new Complex(cClone.getReal(), -cClone.getImag()), c);
        
        d.mutateConjugate();
        assertEquals(new Complex(dClone.getReal(), -dClone.getImag()), d);
    }

    /**
     * Test of getConjugate method, of class Complex.
     */
    @Test
    public void testGetConjugate()
    {
        System.out.println("getConjugate");
        Complex result = a.getConjugate();
        assertEquals(aClone, a);
        assertEquals(new Complex(aClone.getReal(), -aClone.getImag()), result);
        
        result = c.getConjugate();
        assertEquals(bClone, b);
        assertEquals(new Complex(cClone.getReal(), -cClone.getImag()), result);
        
        result = d.getConjugate();
        assertEquals(bClone, b);
        assertEquals(new Complex(dClone.getReal(), -dClone.getImag()), result);
    }


    /**
     * Test of clone method, of class Complex.
     */
    @Test
    public void testClone()
    {
        System.out.println("clone");
        Complex clone = a.clone();
        assertEquals(aClone, a);
        assertEquals(aClone, clone);
        clone.setReal(100);
        assertEquals(aClone, a);
        assertTrue(!clone.equals(aClone));
        
    }
}
