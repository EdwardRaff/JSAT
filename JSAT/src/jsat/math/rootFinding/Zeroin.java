
package jsat.math.rootFinding;

import jsat.math.Function;
import static java.lang.Math.*;
/**
 *
 * @author Edward Raff
 */
public class Zeroin
{
    /**
     * Performs a root finding search on the function f(x) = s.
     * 
     * @param a the minimum value in the range to look for the solution
     * @param b the maximum value in the range to look for the solution
     * @param f the function to use
     * @param args the first value of this function must be the value s, 
     * that we want to f(x) = s. All subsiquent arguments will be passed
     * to the function f in given order, and the first argument will be 
     * adjusted. Ie: the search is for f(x, v1, v2, v3 ..) = s, where 
     * all v will be heald constant.
     * 
     * @return the value x, such that f(x) = s
     * @throws ArithmeticException if the desired value is not in the given range 
     */
    public static double root(double a, double b, Function f, double... args)
    {
        if(args.length < 1)
            throw new ArithmeticException("Bisection method requires a value to search for");
        
        
        /*
         * Code has few comments, taken fro algorithum descriptoin http://en.wikipedia.org/wiki/Brent%27s_method#Algorithm ,
         * which is from Brent's book (according to comments, I would like to get the book either way)
         * 
         */
        
        /**
         * The shift value, the method finds the root f(x) = 0, we want f(x) = s, so we solve f(x)-s) = 0
         */
        double shift = args[0];
   
        args[0] = a;
        double fa = f.f(args)-shift;
        args[0] = b;
        double fb = f.f(args)-shift;
        
        if(fa * fb >= 0)
            throw new ArithmeticException("The given search interval doe snot appear to contain the desired root " + shift);
        
        if(abs(fa) < abs(fb)) //swap
        {
            double tmp = a;
            a = b;
            b = tmp;
            
            tmp = fa;
            fa = fb;
            fb = tmp;
        }
        
        
        double c = a;
        double fc = fa;
        boolean mflag = true;
        double s;
        double d = 0;//inital value dosnt matter, and will not be used
        
        double fs;
        
        do
        {
            if(fa != fc && fb != fc)//inverse quadratic interpolation
            {
                s = a*fb*fc/( (fa-fb)*(fa-fc) ) + b*fa*fc/( (fb-fa)*(fb-fc) ) + c*fa*fb/( (fc-fa)*(fc-fb) );
            }
            else//secant rule
            {
                s = b - fb*(b-a)/(fb-fa);
            }
            
            
            //Determin wethor or not we must use bisection
            
            boolean cond1 = (s - ( 3 * a + b) / 4 ) * ( s - b) >= 0;
            boolean cond2 = mflag && (abs(s - b) >= (abs(b - c) / 2));
            boolean cond3 = !mflag && (abs(s - b) >= (abs(c - d) / 2));
            boolean cond4 = mflag && (abs(b-c) < 2e-15);
            boolean cond5 = !mflag && abs(c-d) < 2e-15;
            
            if(cond1 || cond2 || cond3 || cond4 || cond5)//Bisection must be used
            {
                s = (a+b)/2;
                mflag = true;
            }
            else
                mflag = false;
            
            args[0] = s;
            fs = f.f(args)-shift;
            d = c;
            c = b;
            
            //adjust the interval accordingly
            if(fa*fs < 0)
            {
                b = s;
                fb = fs;
            }
            else
            {
                a = s;
                fa = fs;
            }
            
            if(abs(fa) < abs(fb))//swap
            {
                double tmp = a;
                a = b;
                b = tmp;
                
                tmp = fa;
                fa = fb;
                fb = tmp;
                
            }

        }
        while( fb != 0.0 && fs != 0.0 && abs(b-a) > 2e-15);
        
        
        return b;
    }
}
