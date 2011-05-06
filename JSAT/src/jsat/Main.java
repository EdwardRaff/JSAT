
package jsat;

import jsat.math.rootFinding.Secant;
import jsat.distributions.NormalDistribution;
import jsat.math.ContinuedFraction;
import jsat.math.Function;
import jsat.math.SpecialMath;
import jsat.math.integration.Romberg;
import jsat.math.integration.Trapezoidal;
import jsat.math.rootFinding.Bisection;
import static java.lang.Math.*;
import static jsat.math.SpecialMath.*;

/**
 *
 * @author Edward Raff
 */
public class Main {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args)
    {
        // TODO code application logic here

        Function func = new Function() {

            public double f(double... x)
            {
                return Math.cos(x[0]);
            }
        };
//        System.out.println(Trapezoidal.trapz(func, 0, 1, 1000));
//        System.out.println(Romberg.romb(func, 0, 1, 20));
//        System.out.println("0.6321189695429691");
//        System.out.println(0.8427007929497149);
        
        
        ContinuedFraction cf2 = new ContinuedFraction() {

            @Override
            public double getA(int pos, double... args)
            {

                if(pos % 2 == 0)
                {
                    pos /= 2;//the # of the even term
                    
                    return pos*args[1];
                }
                else
                {
                    pos/=2;
                    
                    return -(args[0]+pos)*args[1];
                }
            }

            @Override
            public double getB(int pos, double... args)
            {
                
                return args[0];
            }
        };
        
                
//        double a = 0.5;
//        double b = 5;
//        double x = 0.025;
//        double numer = a*log(x)+b*log(1-x)-log(a)-lnBeta(a, b);
//        
//        double z = 1.3;
//        
//        System.out.println(0.10686371499337947);
//        System.out.println(exp(a*log(z)-z-lnGamma(a))/gammaQ.lentz(a, z));
//        System.out.println(exp(a*log(z)-z-lnGamma(a))/gammaQ.lentzE(x,a,b));
//        System.out.println(exp(a*log(z)-z-lnGamma(a))/gammaQ.lentzO(x,a,b));
//        System.out.println((exp(a*log(z)-z-lnGamma(a))/gammaQ.lentzO(x,a,b) + exp(a*log(z)-z-lnGamma(a))/gammaQ.lentzE(x,a,b))/2);
//        System.out.println(exp(a*log(z)-z-lnGamma(a))/gammaQ.lentzBackward(x,a,b));

//        Bisection.root(0, 100, func, 2.0);
        System.out.println(Secant.root(1, Math.PI, func, 0.2));
        

//        for(double x = 0.025; x <= 1; x+=0.025)
//        {
////            System.out.print(SpecialMath.lnLowIncGamma(a, x) + ","); 296
//            System.out.print(Double.toString(SpecialMath.betaIncReg(x, 80, 100)).replaceAll("E", "*10^") + ",");
////            System.out.print(x + ",");
//            
//        }
        System.out.println();


//        System.out.println(SpecialMath.lnLowIncGamma1(a, 41.5));
        
        System.out.println();
        
//        System.out.println(Math.exp(SpecialMath.lnLowIncGamma(a, a)));
//        System.out.println(3397585.145212605);
//        System.out.println(Math.pow(a, a)*Math.exp(-a)/cf.lentz(a, a));
//        System.out.println(Math.pow(a, a)*Math.exp(-a)/cf.backwardNaive(800, a, a));
        
        
        
    }

}
