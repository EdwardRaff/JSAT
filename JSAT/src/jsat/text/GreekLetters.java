
package jsat.text;

/**
 *
 * @author Edward Raff
 */
public class GreekLetters
{
    public static final String alpha = "\u03B1";
    public static final String beta = "\u03B2";
    public static final String gamma = "\u03B3";
    public static final String delta = "\u03B4";
    public static final String epsilon = "\u03B5";
    public static final String zeta = "\u03B6";
    public static final String eta = "\u03B7";
    public static final String theta = "\u03B8";
    public static final String iota = "\u03B9";
    public static final String kappa = "\u03BA";
    public static final String lamda = "\u03BB";
    public static final String mu = "\u03BC";
    public static final String nu = "\u03BD";
    public static final String xi = "\u03BE";
    public static final String omicron = "\u03BF";
    public static final String pi = "\u03C0";
    public static final String rho = "\u03C1";
    public static final String finalSigma = "\u03C2";
    public static final String sigma = "\u03C3";
    public static final String tau = "\u03C4";
    public static final String upsilon = "\u03C5";
    public static final String phi = "\u03C6";
    public static final String chi = "\u03C7";
    public static final String psi = "\u03C8";
    public static final String omega = "\u03C9";
    
    /**
     * Puts an over line on top the string s.
     * @param s the character to put a line over
     * @return the input with a line over
     */
    public static String bar(String s)
    {
        return s + "\u0305";
    }
    
}
