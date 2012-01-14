
package jsat.graphing;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Frame;
import java.awt.GridLayout;
import java.util.List;
import javax.swing.BorderFactory;
import javax.swing.JDialog;
import javax.swing.JLabel;
import javax.swing.JPanel;
import jsat.DataSet;
import jsat.linear.Vec;

/**
 *
 * @author Edward Raff
 */
public class ScatterplotMatrix extends JDialog
{

    public ScatterplotMatrix(Frame parent, String title, DataSet dataSet)
    {
        super(parent, title, false);

        JPanel panel = new JPanel();
        int numerVals = dataSet.getNumNumericalVars();
        panel.setLayout(new GridLayout(numerVals, numerVals));

        for(int i = 0; i < numerVals; i++)
        {
            Vec yAxis = dataSet.getNumericColumn(i);
            for(int j = 0; j < numerVals; j++)
            {
                if(i == j)
                {
                    
                    JLabel tmp = new JLabel(dataSet.getNumericName(i), JLabel.CENTER);
                    tmp.setBorder(BorderFactory.createLineBorder(Color.black));
                    panel.add(tmp);
                    continue;
                }

                Vec xAxis = dataSet.getNumericColumn(j);

                ScatterPlot sp = new ScatterPlot(xAxis, yAxis);
                sp.setPadding(0);
                sp.setBorder(BorderFactory.createLineBorder(Color.black));
                panel.add(sp);

            }
        }

        this.setContentPane(panel);
    }




}
