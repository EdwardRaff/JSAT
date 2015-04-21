package jsat.math;

import java.io.Serializable;

/**
 * A class for representing a complex value by a real and imaginary double pair.
 * 
 * @author Edward Raff
 */
public class Complex implements Cloneable, Serializable
{

	private static final long serialVersionUID = -2219274170047061708L;
	private double real, imag;
    
    /**
     * Returns the complex number representing sqrt(-1)
     * @return the complex number <i>i</i>
     */
    public static Complex I()
    {
        return new Complex(0.0, 1.0);
    }

    /**
     * Creates a new Complex number 
     * @param real the real part of the number 
     * @param imag the imaginary part of the number
     */
    public Complex(double real, double imag)
    {
        this.real = real;
        this.imag = imag;
    }

    /**
     * Returns the real part of this complex number
     * @return the real part of this complex number
     */
    public double getReal()
    {
        return real;
    }
    
    /**
     * Sets the real value part of this complex number
     * @param r the new real value
     */
    public void setReal(double r)
    {
        this.real = r;
    }

    /**
     * Sets the imaginary value part of this complex number
     * @param imag the new imaginary value
     */
    public void setImag(double imag)
    {
        this.imag = imag;
    }
    

    /**
     * Returns the imaginary part of this complex number
     * @return the imaginary part of this complex number
     */
    public double getImag()
    {
        return imag;
    }
    
    /**
     * Alters this complex number as if an addition of another complex number was performed. 
     * @param r the real part of the other number
     * @param i the imaginary part of the other number
     */
    public void mutableAdd(double r, double i)
    {
        this.real += r;
        this.imag += i;
    }
    
    /**
     * Alters this complex number to contain the result of the addition of another
     * @param c the complex value to add to this
     */
    public void mutableAdd(Complex c)
    {
        mutableAdd(c.real, c.imag);
    }
    
    /**
     * Creates a new complex number containing the resulting addition of this and another
     * @param c the number to add
     * @return <tt>this</tt>+c
     */
    public Complex add(Complex c)
    {
        Complex ret = new Complex(real, imag);
        ret.mutableAdd(c);
        return ret;
    }
    
    /**
     * Alters this complex number as if a subtraction of another complex number was performed. 
     * @param r the real part of the other number
     * @param i the imaginary part of the other number
     */
    public void mutableSubtract(double r, double i)
    {
        mutableAdd(-r, -i);
    }
    
    /**
     * Alters this complex number to contain the result of the subtraction of another
     * @param c the number to subtract
     */
    public void mutableSubtract(Complex c)
    {
        mutableSubtract(c.real, c.imag);
    }
    
    /**
     * Creates a new complex number containing the resulting subtracting another from this one
     * @param c the number to subtract
     * @return <tt>this</tt>-c
     */
    public Complex subtract(Complex c)
    {
        Complex ret = new Complex(real, imag);
        ret.mutableSubtract(c);
        return ret;
    }
    
    /**
     * Performs a complex multiplication
     * 
     * @param a the real part of the first number
     * @param b the imaginary part of the first number
     * @param c the real part of the second number
     * @param d the imaginary part of the second number 
     * @param results an array to store the real and imaginary results in. First index is the real, 2nd is the imaginary. 
     */
    public static void cMul(double a, double b, double c, double d, double[] results)
    {
        results[0] = a*c-b*d;
        results[1] = b*c+a*d;
    }
    
    /**
     * Alters this complex number as if a multiplication of another complex number was performed. 
     * @param c the real part of the other number
     * @param d the imaginary part of the other number
     */
    public void mutableMultiply(double c, double d)
    {
        double newR = this.real*c-this.imag*d;
        double newI = this.imag*c+this.real*d;
        this.real = newR;
        this.imag = newI;
    }
    
    /**
     * Alters this complex number to contain the result of the multiplication of another
     * @param c the number to multiply by
     */
    public void mutableMultiply(Complex c)
    {
        mutableMultiply(c.real, c.imag);
    }
    
    /**
     * Creates a new complex number containing the resulting multiplication between this and another
     * @param c the number to multiply by
     * @return <tt>this</tt>*c
     */
    public Complex multiply(Complex c)
    {
        Complex ret = new Complex(real, imag);
        ret.mutableMultiply(c);
        return ret;
    }
    
    /**
     * Performs a complex division operation. <br>
     * The standard complex division performs a set of operations that is 
     * suseptible to both overflow and underflow. This method is more 
     * numerically stable while still being relatively fast to execute. 
     * 
     * @param a the real part of the first number
     * @param b the imaginary part of the first number
     * @param c the real part of the second number
     * @param d the imaginary part of the second number 
     * @param results an array to store the real and imaginary results in. First
     * index is the real, 2nd is the imaginary. 
     */
    public static void cDiv(double a, double b, double c, double d, double[] results)
    {
        /**
         * Douglas M. Priest. Efficient scaling for complex division. ACM Trans. 
         * Math. Softw., 30(4):389–401, 2004
         */
        long aa, bb, cc, dd, ss;
        double t;
        int ha, hb, hc, hd, hz, hw, hs;
        
        /*extract high-order 32 bits to estimate |z| and |w| */
        aa = Double.doubleToRawLongBits(a);
        bb = Double.doubleToRawLongBits(b);
        
        ha = (int) ((aa >> 32) & 0x7fffffff);
        hb = (int) ((bb >> 32) & 0x7fffffff);
        hz = (ha > hb)? ha : hb;
        
        cc = Double.doubleToRawLongBits(c);
        dd = Double.doubleToRawLongBits(d);
        
        hc = (int) ((cc >> 32) & 0x7fffffff);
        hd = (int) ((dd >> 32) & 0x7fffffff);
        hw = (hc > hd)? hc : hd;
        
        /* compute the scale factor */
        if (hz < 0x07200000 && hw >= 0x32800000 && hw < 0x47100000)
        {
            /* |z| < 2^-909 and 2^-215 <= |w| < 2^114 */
            hs = (((0x47100000 - hw) >> 1) & 0xfff00000) + 0x3ff00000;
        }
        else
            hs = (((hw >> 2) - hw) + 0x6fd7ffff) & 0xfff00000;
        ss = ((long) hs) << 32;
        
        /* scale c and d, and compute the quotient */
        double ssd = Double.longBitsToDouble(ss);
        c *= ssd;
        d *= ssd;
        t = 1.0 / (c * c + d * d);
        c *= ssd;
        d *= ssd;
        results[0] = (a * c + b * d) * t;
        results[1] = (b * c - a * d) * t;
    }
    
    /**
     * Alters this complex number as if a division by another complex number was performed. 
     * @param c the real part of the other number
     * @param d the imaginary part of the other number
     */
    public void mutableDivide(double c, double d)
    {
        final double[] r = new double[2];
        cDiv(real, imag, c, d, r);
        this.real = r[0];
        this.imag = r[1];
    }
    
    /**
     * Alters this complex number to contain the result of the division by another
     * @param c the number to divide by
     */
    public void mutableDivide(Complex c)
    {
        mutableDivide(c.real, c.imag);
    }
    
    /**
     * Creates a new complex number containing the resulting division of this by
     * another
     * 
     * @param c the number to divide by
     * @return <tt>this</tt>/c
     */
    public Complex divide(Complex c)
    {
        Complex ret = new Complex(real, imag);
        ret.mutableDivide(c);
        return ret;
    }

    /**
     * Computes the magnitude of this complex number, which is 
     * sqrt({@link #getReal() Re}<sup>2</sup>+{@link #getImag() Im}<sup>2</sup>)
     * @return the magnitude of this complex number
     */
    public double getMagnitude()
    {
        return Math.hypot(real, imag);
    }
    
    /**
     * Computes the Argument, also called phase, of this complex number. Unless 
     * the result is {@link Double#NaN}, which occurs only for complex zero, the
     * result will be in the range [-pi, pi]
     * @return the argument of this complex number
     */
    public double getArg()
    {
        return Math.atan2(imag, real);
    }
    
    /**
     * Alters this complex number so that it represents its complex conjugate 
     * instead. 
     */
    public void mutateConjugate()
    {
        this.imag = -imag;
    }
    
    /**
     * Returns a new complex number representing the complex conjugate of this 
     * one
     * @return the complex conjugate of <tt>this</tt>
     */
    public Complex getConjugate()
    {
        return new Complex(real, -imag);
    }
    
    @Override
    public String toString()
    {
        if(imag == 0)
            return Double.toString(real);
        else if(real == 0)
            return imag + "i";
        else
            return "("+real + " + " + imag + "i)";
    }

    @Override
    protected Complex clone()
    {
        return new Complex(real, imag);
    }

    @Override
    public boolean equals(Object obj)
    {
        return equals(obj, 0.0);
    }

    @Override
    public int hashCode()
    {
        int hash = 5;
        hash = 67 * hash + (int) (Double.doubleToLongBits(this.real) ^ (Double.doubleToLongBits(this.real) >>> 32));
        hash = 67 * hash + (int) (Double.doubleToLongBits(this.imag) ^ (Double.doubleToLongBits(this.imag) >>> 32));
        return hash;
    }

    /**
     * Checks if <i>this</i> is approximately equal to another Complex object
     * @param obj the object to compare against
     * @param eps the maximum acceptable difference between values to be 
     * considered equal
     * @return <tt>true</tt> if the objects are approximately equal
     */
    public boolean equals(Object obj, double eps)
    {
        if( obj instanceof Complex)
        {
            Complex other = (Complex) obj;
            if(Math.abs(this.real-other.real) > eps)
                return false;
            else if(Math.abs((this.imag - other.imag)) > eps)
                return false;
            return true;
        }
        return false;
    }
    
    
}
