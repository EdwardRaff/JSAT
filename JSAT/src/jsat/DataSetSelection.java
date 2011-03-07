
package jsat;

import java.awt.BorderLayout;
import java.awt.Frame;
import java.awt.GridBagLayout;
import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.beans.PropertyChangeListener;
import javax.swing.Action;
import javax.swing.BorderFactory;
import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JDialog;
import javax.swing.JList;
import javax.swing.JPanel;
import javax.swing.JScrollPane;

/**
 *
 * @author Edward Raff
 */
public class DataSetSelection extends JDialog
{

    final String[] dataSelections;
    final String[] reasons;
    JComboBox[] boxs;

    public DataSetSelection(Frame parent, String title, String[] dataSelections, String[] reasons)
    {
        super(parent, title, true);
        this.dataSelections = dataSelections;
        this.reasons = reasons;
        boxs = new JComboBox[reasons.length];
        
        JPanel optionPanel = new JPanel(new GridLayout(dataSelections.length, 1));
        JPanel fullPanel = new JPanel(new BorderLayout());
        


        for(int i = 0; i < reasons.length; i++)
        {
            final JComboBox jc = new JComboBox(dataSelections);
            jc.setBorder(BorderFactory.createTitledBorder(reasons[i]));

            boxs[i] = jc;
            optionPanel.add(jc);
        }

        fullPanel.add(new JScrollPane(optionPanel), BorderLayout.CENTER);


        JButton closeButton = new JButton("Ok");
        closeButton.addActionListener(new ActionListener() {

            public void actionPerformed(ActionEvent e)
            {
                setVisible(false); 
            }
        });

        fullPanel.add(closeButton, BorderLayout.SOUTH);

        setContentPane(fullPanel);

    }

    public int[] getSelections()
    {
        int[] selections = new int[reasons.length];
        pack();
        setVisible(true);
        for(int i = 0; i < reasons.length; i++)
            selections[i] = boxs[i].getSelectedIndex();

        return selections;
    }


}
