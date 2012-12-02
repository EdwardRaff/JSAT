package jsat.guitool;

import java.awt.Color;
import java.util.ArrayList;
import java.util.List;

/**
 * Provides some simple utility methods that may aid in developing GUIs 
 * 
 * @author Edward Raff
 */
public class GUIUtils
{

    private GUIUtils()
    {
    }
    
    /**
     * Computes several visually distinct colors such that they are equally 
     * spaced apart in the HSB color space. 
     * 
     * @param k how many colors to get
     * @return a list of colors
     */
    public static List<Color> getDistinctColors(int k)
    {
        return getDistinctColors(k, 0.95f, 0.7f);
    }

    /**
     * Computes several visually distinct colors such that they are equally 
     * spaced apart in the HSB color space. 
     * 
     * @param k how many colors to get
     * @param saturation the saturation level for all the colors
     * @param brightness the brightness for all the colors
     * @return a list of colors
     */
    public static List<Color> getDistinctColors(int k, float saturation, float brightness)
    {
        ArrayList<Color> categoryColors = new ArrayList<Color>(k);
        float colorFactor = 1.0f / k;
        for (int i = 0; i < k; i++)
        {
            Color c = Color.getHSBColor(i * colorFactor, saturation, brightness);
            categoryColors.add(c);
        }

        return categoryColors;
    }
}
