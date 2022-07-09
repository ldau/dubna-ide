# -*- coding: utf-8 -*-
#
# Copyright © Spyder Project Contributors
# Licensed under the terms of the MIT License
# (see spyder/__init__.py for details)

"""
General entry in Preferences.

For historical reasons (dating back to Spyder 2) the main class here is called
`MainConfigPage` and its associated entry in our config system is called
`main`.
"""

import traceback
import sys

from qtpy.compat import from_qvariant
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (QApplication, QButtonGroup, QGridLayout, QGroupBox,
                            QHBoxLayout, QLabel, QMessageBox, QTabWidget,
                            QVBoxLayout, QWidget)

from spyder.config.base import (_, DISABLED_LANGUAGES, LANGUAGE_CODES,
                                running_in_mac_app, save_lang_conf)
from spyder.api.preferences import PluginConfigPage
from spyder.py3compat import to_text_string

HDPI_QT_PAGE = "https://doc.qt.io/qt-5/highdpi.html"


class ApplicationConfigPage(PluginConfigPage):
    APPLY_CONF_PAGE_SETTINGS = True

    def setup_page(self):
        newcb = self.create_checkbox

        # --- Advanced
        # Remove disabled languages
        language_codes = LANGUAGE_CODES.copy()
        for lang in DISABLED_LANGUAGES:
            language_codes.pop(lang)

        languages = language_codes.items()
        language_choices = sorted([(val, key) for key, val in languages])
        language_combo = self.create_combobox(_('Language:'),
                                              language_choices,
                                              'interface_language',
                                              restart=True)

        opengl_options = ['Automatic', 'Desktop', 'Software', 'GLES']
        opengl_choices = list(zip(opengl_options,
                                  [c.lower() for c in opengl_options]))
        opengl_combo = self.create_combobox(_('Rendering engine:'),
                                            opengl_choices,
                                            'opengl',
                                            restart=True)

        single_instance_box = newcb(_("Use a single instance"),
                                    'single_instance',
                                    tip=_("Set this to open external<br> "
                                          "Python files in an already running "
                                          "instance (Requires a restart)"))

        prompt_box = newcb(_("Prompt when exiting"), 'prompt_on_exit')
        popup_console_box = newcb(_("Show internal Spyder errors to report "
                                    "them to Github"), 'show_internal_errors')
        check_updates = newcb(_("Check for updates on startup"),
                              'check_updates_on_startup')

        # Decide if it's possible to activate or not single instance mode
        if running_in_mac_app():
            self.set_option("single_instance", True)
            single_instance_box.setEnabled(False)

        comboboxes_advanced_layout = QHBoxLayout()
        cbs_adv_grid = QGridLayout()
        cbs_adv_grid.addWidget(language_combo.label, 0, 0)
        cbs_adv_grid.addWidget(language_combo.combobox, 0, 1)
        cbs_adv_grid.addWidget(opengl_combo.label, 1, 0)
        cbs_adv_grid.addWidget(opengl_combo.combobox, 1, 1)
        comboboxes_advanced_layout.addLayout(cbs_adv_grid)
        comboboxes_advanced_layout.addStretch(1)

        advanced_layout = QVBoxLayout()
        advanced_layout.addLayout(comboboxes_advanced_layout)
        advanced_layout.addWidget(single_instance_box)
        advanced_layout.addWidget(prompt_box)
        advanced_layout.addWidget(popup_console_box)
        advanced_layout.addWidget(check_updates)

        advanced_widget = QWidget()
        advanced_widget.setLayout(advanced_layout)

        # --- Panes
        interface_group = QGroupBox(_("Panes"))

        verttabs_box = newcb(_("Vertical tabs in panes"),
                             'vertical_tabs')
        margin_box = newcb(_("Custom margin for panes:"),
                           'use_custom_margin')
        margin_spin = self.create_spinbox("", _("pixels"), 'custom_margin',
                                          default=0, min_=0, max_=30)
        margin_box.toggled.connect(margin_spin.spinbox.setEnabled)
        margin_box.toggled.connect(margin_spin.slabel.setEnabled)
        margin_spin.spinbox.setEnabled(self.get_option('use_custom_margin'))
        margin_spin.slabel.setEnabled(self.get_option('use_custom_margin'))

        cursor_box = newcb(_("Cursor blinking:"),
                           'use_custom_cursor_blinking')
        cursor_spin = self.create_spinbox(
            "", _("ms"),
            'custom_cursor_blinking',
            default=QApplication.cursorFlashTime(),
            min_=0, max_=5000, step=100)
        cursor_box.toggled.connect(cursor_spin.spinbox.setEnabled)
        cursor_box.toggled.connect(cursor_spin.slabel.setEnabled)
        cursor_spin.spinbox.setEnabled(
            self.get_option('use_custom_cursor_blinking'))
        cursor_spin.slabel.setEnabled(
            self.get_option('use_custom_cursor_blinking'))

        margins_cursor_layout = QGridLayout()
        margins_cursor_layout.addWidget(margin_box, 0, 0)
        margins_cursor_layout.addWidget(margin_spin.spinbox, 0, 1)
        margins_cursor_layout.addWidget(margin_spin.slabel, 0, 2)
        margins_cursor_layout.addWidget(cursor_box, 1, 0)
        margins_cursor_layout.addWidget(cursor_spin.spinbox, 1, 1)
        margins_cursor_layout.addWidget(cursor_spin.slabel, 1, 2)
        margins_cursor_layout.setColumnStretch(2, 100)

        # Layout interface
        interface_layout = QVBoxLayout()
        interface_layout.addWidget(verttabs_box)
        interface_layout.addLayout(margins_cursor_layout)
        interface_group.setLayout(interface_layout)

        if sys.platform == "darwin" and not running_in_mac_app():
            # To open files from Finder directly in Spyder.
            from spyder.utils.qthelpers import (register_app_launchservices,
                                                restore_launchservices)
            import applaunchservices as als

            def set_open_file(state):
                if state:
                    register_app_launchservices()
                else:
                    restore_launchservices()

            macOS_group = QGroupBox(_("macOS integration"))
            mac_open_file_box = newcb(
                _("Open files from Finder with Spyder"),
                'mac_open_file',
                tip=_("Register Spyder with the Launch Services"))
            mac_open_file_box.toggled.connect(set_open_file)
            macOS_layout = QVBoxLayout()
            macOS_layout.addWidget(mac_open_file_box)
            if als.get_bundle_identifier() is None:
                # Disable setting
                mac_open_file_box.setDisabled(True)
                macOS_layout.addWidget(QLabel(
                    _('Launch Spyder with <code>python.app</code> to enable'
                      ' Apple event integrations.')))

            macOS_group.setLayout(macOS_layout)

        # --- Screen resolution Group (hidpi)
        screen_resolution_group = QGroupBox(_("Screen resolution"))
        screen_resolution_bg = QButtonGroup(screen_resolution_group)
        screen_resolution_label = QLabel(_("Configuration for high DPI "
                                           "screens<br><br>"
                                           "Please see "
                                           "<a href=\"{0}\">{0}</a><> "
                                           "for more information about "
                                           "these options (in "
                                           "English).").format(HDPI_QT_PAGE))
        screen_resolution_label.setWordWrap(True)
        screen_resolution_label.setOpenExternalLinks(True)

        normal_radio = self.create_radiobutton(
                                _("Normal"),
                                'normal_screen_resolution',
                                button_group=screen_resolution_bg)
        auto_scale_radio = self.create_radiobutton(
                                _("Enable auto high DPI scaling"),
                                'high_dpi_scaling',
                                button_group=screen_resolution_bg,
                                tip=_("Set this for high DPI displays"),
                                restart=True)

        custom_scaling_radio = self.create_radiobutton(
                                _("Set a custom high DPI scaling"),
                                'high_dpi_custom_scale_factor',
                                button_group=screen_resolution_bg,
                                tip=_("Set this for high DPI displays when "
                                      "auto scaling does not work"),
                                restart=True)

        self.custom_scaling_edit = self.create_lineedit(
            "",
            'high_dpi_custom_scale_factors',
            tip=_("Enter values for different screens "
                  "separated by semicolons ';'.\n"
                  "Float values are supported"),
            alignment=Qt.Horizontal,
            regex=r"[0-9]+(?:\.[0-9]*)(;[0-9]+(?:\.[0-9]*))*",
            restart=True)

        normal_radio.toggled.connect(self.custom_scaling_edit.setDisabled)
        auto_scale_radio.toggled.connect(self.custom_scaling_edit.setDisabled)
        custom_scaling_radio.toggled.connect(
            self.custom_scaling_edit.setEnabled)

        # Layout Screen resolution
        screen_resolution_layout = QVBoxLayout()
        screen_resolution_layout.addWidget(screen_resolution_label)

        screen_resolution_inner_layout = QGridLayout()
        screen_resolution_inner_layout.addWidget(normal_radio, 0, 0)
        screen_resolution_inner_layout.addWidget(auto_scale_radio, 1, 0)
        screen_resolution_inner_layout.addWidget(custom_scaling_radio, 2, 0)
        screen_resolution_inner_layout.addWidget(
            self.custom_scaling_edit, 2, 1)

        screen_resolution_layout.addLayout(screen_resolution_inner_layout)
        screen_resolution_group.setLayout(screen_resolution_layout)
        if sys.platform == "darwin" and not running_in_mac_app():
            interface_tab = self.create_tab(screen_resolution_group,
                                            interface_group, macOS_group)
        else:
            interface_tab = self.create_tab(screen_resolution_group,
                                            interface_group)

        self.tabs = QTabWidget()
        self.tabs.addTab(interface_tab, _("Interface"))
        self.tabs.addTab(self.create_tab(advanced_widget),
                         _("Advanced settings"))

        vlayout = QVBoxLayout()
        vlayout.addWidget(self.tabs)
        self.setLayout(vlayout)

    def apply_settings(self, options):
        if 'high_dpi_custom_scale_factors' in options:
            scale_factors = self.get_option(
                'high_dpi_custom_scale_factors').split(';')
            change_min_scale_factor = False
            for idx, scale_factor in enumerate(scale_factors[:]):
                scale_factor = float(scale_factor)
                if scale_factor < 1.0:
                    change_min_scale_factor = True
                    scale_factors[idx] = "1.0"
            if change_min_scale_factor:
                scale_factors_text = ";".join(scale_factors)
                QMessageBox.critical(
                    self, _("Error"),
                    _("We're sorry but setting a scale factor bellow 1.0 "
                      "isn't possible. Any scale factor bellow 1.0 will "
                      "be set to 1.0"),
                    QMessageBox.Ok)
                self.custom_scaling_edit.textbox.setText(scale_factors_text)
                self.set_option(
                    'high_dpi_custom_scale_factors', scale_factors_text)
                self.changed_options.add('high_dpi_custom_scale_factors')
        self.plugin.apply_settings()

    def _save_lang(self):
        """
        Get selected language setting and save to language configuration file.
        """
        for combobox, (sec, opt, _default) in list(self.comboboxes.items()):
            if opt == 'interface_language':
                data = combobox.itemData(combobox.currentIndex())
                value = from_qvariant(data, to_text_string)
                break
        try:
            save_lang_conf(value)
            self.set_option('interface_language', value)
        except Exception:
            QMessageBox.critical(
                self, _("Error"),
                _("We're sorry but the following error occurred while trying "
                  "to set your selected language:<br><br>"
                  "<tt>{}</tt>").format(traceback.format_exc()),
                QMessageBox.Ok)
            return
