
package jsat.math.rootFinding;

import jsat.math.Function;
import static java.lang.Math.*;

/**
 *
 * @author Edward Raff
 */
public class RiddersMethod
{
    public static double root(double x1, double x2, Function f, double... args)
    {
        if(args.length < 1)
            throw new ArithmeticException("Bisection method requires a value to search for");
  
        double s = args[0];
        
        args[0] = x1;
        double fx1 = f.f(args)-s;
        args[0] = x2;
        double fx2 = f.f(args)-s;
        
        if(fx1* fx2 >= 0)
            throw new ArithmeticException("The given interval does not appear to bracket the root");
        
        int k = 0;
        double dif = 1;//Measure the change interface values
        while( abs(x1-x2) > 1e-15)
        {
            double x3 = (x1+x2)/2;
            
            args[0] = x3;
            double fx3 = f.f(args)-s;
            
            double x4 = x3+(x3-x1)*signum(fx1-fx2)*fx3/sqrt(fx3*fx3-fx1*fx2); 
         
            args[0] = x4;
            double fx4 = f.f(args)-s;
            
            if(fx1 == fx4)
            {
                return x4;
            }
            else if(fx1 * fx4 < 0)
            {
                dif = abs(x4 - x2);
                if(dif == 0)//WE are no longer updating, return the value
                    return x4;
                x2 = x4;
                fx2 = fx4;
            }
            else if(fx2 == fx4)
            {
                return x4;
            }
            else
            {
                dif = abs(x4 - x1);
                if(dif == 0)//WE are no longer updating, return the value
                    return x4;
                x1 = x4;
                fx1 = fx4;
            }
            
        }
        
        
        return x2;
    }
}
