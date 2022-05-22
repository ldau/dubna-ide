# -*- coding: utf-8 -*-
#
# Copyright © 2018 André Roberge - mod_pydoc
# Copyright © Spyder Project Contributors
# Licensed under the terms of the MIT License
# (see spyder/__init__.py for details)

"""PyDoc patch"""
# Standard libray
import asyncio
import builtins
import io
import inspect
import os
import pkgutil
import platform
import re
import sys
import tokenize
import warnings


# Local imports
import tornado

from spyder.config.base import _, DEV
from spyder.config.gui import is_dark_interface, get_font
from spyder.py3compat import PY2, to_text_string


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("Hello, world")


class MainHandler2(tornado.web.RequestHandler):
    def get(self):
        self.write(open(r'C:\Users\dovran\dubna-ide\example.py').read())


if not PY2:
    from pydoc import (
        classname, classify_class_attrs, describe, Doc, format_exception_only,
        Helper, HTMLRepr, _is_bound_method, ModuleScanner, locate, replace,
        visiblename, isdata, getdoc, deque, _split_list)



def _url_handler(url, content_type="text/html"):
    """Pydoc url handler for use with the pydoc server.

    If the content_type is 'text/css', the _pydoc.css style
    sheet is read and returned if it exits.

    If the content_type is 'text/html', then the result of
    get_html_page(url) is returned.

    See https://github.com/python/cpython/blob/master/Lib/pydoc.py
    """
    class _HTMLDoc(CustomHTMLDoc):

        def page(self, title, contents):
            """Format an HTML page."""
            rich_text_font = get_font(option="rich_font").family()
            plain_text_font = get_font(option="font").family()

            if is_dark_interface():
                css_path = "static/css/dark_pydoc.css"
            else:
                css_path = "static/css/light_pydoc.css"

            css_link = (
                '<link rel="stylesheet" type="text/css" href="/%s">' %
                css_path)

            code_style = (
                '<style>code {font-family: "%s"}</style>' % plain_text_font)

            html_page = '''\
<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 4.0 Transitional//EN">
<html><head><title>Pydoc: %s</title>
<meta http-equiv="Content-Type" content="text/html; charset=utf-8">
%s%s</head><body style="clear:both;font-family:'%s'">
%s<div style="clear:both;padding-top:.7em;">%s</div>
</body></html>''' % (title, css_link, code_style, rich_text_font,
                     html_navbar(), contents)

            return html_page

        def filelink(self, url, path):
            return '<a href="getfile?key=%s">%s</a>' % (url, path)

    html = _HTMLDoc()

    def html_navbar():
        version = html.escape("%s [%s, %s]" % (platform.python_version(),
                                               platform.python_build()[0],
                                               platform.python_compiler()))
        return """
            <div style='float:left'>
                Python %s<br>%s
            </div>
            <div style='float:right'>
                <div style='text-align:right; padding-bottom:.7em;'>
                  <a href="index.html">Module Index</a>
                  : <a href="topics.html">Topics</a>
                  : <a href="keywords.html">Keywords</a>
                </div>
                <div style='text-align:right;'>
                    <form action="search" style='display:inline;'>
                      <input class="input-search" type=text name=key size="22">
                      <input class="submit-search" type=submit value="Search">
                    </form>
                </div>
            </div>
            """ % (version, html.escape(platform.platform(terse=True)))

    def html_index():
        """Index page."""
        def bltinlink(name):
            return '<a href="%s.html">%s</a>' % (name, name)

        heading = html.heading('<span>Index of Modules</span>')
        names = [name for name in sys.builtin_module_names
                 if name != '__main__']
        contents = html.multicolumn(names, bltinlink)
        contents = [heading, '<p>' + html.bigsection(
            'Built-in Modules', contents, css_class="builtin_modules")]

        seen = {}
        for dir in sys.path:

            contents.append(html.index(dir, seen))

        contents.append(
            '<p class="ka_ping_yee"><strong>pydoc</strong> by Ka-Ping Yee'
            '&lt;ping@lfw.org&gt;</p>')
        return 'Index of Modules', ''.join(contents)

    def html_search(key):
        """Search results page."""
        # scan for modules
        search_result = []

        def callback(path, modname, desc):
            if modname[-9:] == '.__init__':
                modname = modname[:-9] + ' (package)'
            search_result.append((modname, desc and '- ' + desc))

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')  # ignore problems during import
            ModuleScanner().run(callback, key)

        # format page
        def bltinlink(name):
            return '<a href="%s.html">%s</a>' % (name, name)

        results = []
        heading = html.heading('Search Results')

        for name, desc in search_result:
            results.append(bltinlink(name) + desc)
        contents = heading + html.bigsection(
            'key = {}'.format(key), '<br>'.join(results), css_class="search")
        return 'Search Results', contents

    def html_getfile(path):
        """Get and display a source file listing safely."""
        path = path.replace('%20', ' ')
        with tokenize.open(path) as fp:
            lines = html.escape(fp.read())
        body = '<pre>%s</pre>' % lines
        heading = html.heading('File Listing')

        contents = heading + html.bigsection('File: {}'.format(path), body,
                                             css_class="getfile")
        return 'getfile %s' % path, contents

    def html_topics():
        """Index of topic texts available."""
        def bltinlink(name):
            return '<a href="topic?key=%s">%s</a>' % (name, name)

        heading = html.heading('Index of Topics') + '<br>'
        names = sorted(Helper.topics.keys())

        contents = html.multicolumn(names, bltinlink)
        contents = heading + html.bigsection(
            'Topics', contents, css_class="topics")
        return 'Topics', contents

    def html_keywords():
        """Index of keywords."""
        heading = html.heading('Index of Keywords')
        names = sorted(Helper.keywords.keys())

        def bltinlink(name):
            return '<a href="topic?key=%s">%s</a>' % (name, name)

        contents = html.multicolumn(names, bltinlink)
        contents = heading + '<br>' + html.bigsection(
            'Keywords', contents, css_class="keywords")
        return 'Keywords', contents

    def html_topicpage(topic):
        """Topic or keyword help page."""
        buf = io.StringIO()
        htmlhelp = Helper(buf, buf)
        contents, xrefs = htmlhelp._gettopic(topic)
        if topic in htmlhelp.keywords:
            title = 'Keyword'
        else:
            title = 'Topic'
        heading = html.heading(title)
        contents = '<pre>%s</pre>' % html.markup(contents)
        contents = html.bigsection(topic, contents, css_class="topics")
        if xrefs:
            xrefs = sorted(xrefs.split())

            def bltinlink(name):
                return '<a href="topic?key=%s">%s</a>' % (name, name)

            xrefs = html.multicolumn(xrefs, bltinlink)
            xrefs = html.html_section('Related help topics: ', xrefs,
                                      css_class="topics")
        return ('%s %s' % (title, topic),
                ''.join((heading, contents, xrefs)))

    def html_getobj(url):
        obj = locate(url, forceload=1)
        if obj is None and url != 'None':
            raise ValueError(
                _('There was an error while retrieving documentation '
                  'for the object you requested: Object could not be found'))
        title = describe(obj)
        content = html.document(obj, url)
        return title, content

    def html_error(url, exc):
        heading = html.heading('Error')
        if DEV:
            contents = '<br>'.join(html.escape(line) for line in
                                   format_exception_only(type(exc), exc))
        else:
            contents = '%s' % to_text_string(exc)
        contents = heading + html.bigsection(url, contents, css_class="error")
        return "Error - %s" % url, contents

    def get_html_page(url):
        """Generate an HTML page for url."""
        complete_url = url
        if url.endswith('.html'):
            url = url[:-5]
        try:
            if url in ("", "index"):
                title, content = html_index()
            elif url == "topics":
                title, content = html_topics()
            elif url == "keywords":
                title, content = html_keywords()
            elif '=' in url:
                op, _, url = url.partition('=')
                if op == "search?key":
                    title, content = html_search(url)
                elif op == "getfile?key":
                    title, content = html_getfile(url)
                elif op == "topic?key":
                    # try topics first, then objects.
                    try:
                        title, content = html_topicpage(url)
                    except ValueError:
                        title, content = html_getobj(url)
                elif op == "get?key":
                    # try objects first, then topics.
                    if url in ("", "index"):
                        title, content = html_index()
                    else:
                        try:
                            title, content = html_getobj(url)
                        except ValueError:
                            title, content = html_topicpage(url)
                else:
                    raise ValueError(
                        _('There was an error while retrieving documentation '
                          'for the object you requested: Bad URL %s') % url)
            else:
                title, content = html_getobj(url)
        except Exception as exc:
            # Catch any errors and display them in an error page.
            title, content = html_error(complete_url, exc)
        return html.page(title, content)

    if url.startswith('/'):
        url = url[1:]
    if content_type == 'text/css':
        path_here = os.path.dirname(os.path.realpath(__file__))
        css_path = os.path.join(path_here, url)
        with open(css_path) as fp:
            return ''.join(fp.readlines())
    elif content_type == 'text/html':
        return get_html_page(url)
    # Errors outside the url handler are caught by the server.
    raise TypeError(
        _('There was an error while retrieving documentation '
          'for the object you requested: unknown content type %r for url %s')
          % (content_type, url))


def _start_server(hostname, port):
    """
    Start an HTTP server thread on a specific port.

    This is a reimplementation of `pydoc._start_server` to handle connection
    errors for 'do_GET'.

    Taken from PyDoc: https://github.com/python/cpython/blob/3.7/Lib/pydoc.py
    """
    import http.server
    import email.message
    import select
    import threading
    import time

    class ServerThread(threading.Thread):

        def __init__(self, host, port):
            self.host = host
            self.port = int(port)
            threading.Thread.__init__(self)
            self.serving = False
            self.error = None

        def run(self):
            """Start the server."""
            try:
                asyncio.set_event_loop(asyncio.new_event_loop())
                self.application = tornado.web.Application([
                    (r"/", MainHandler),
                    (r"/src", MainHandler2),
                ])
                self.application.listen(port=self.port, address=self.host)
                self.docserver = tornado.ioloop.IOLoop.current()
                self.serving = True
                self.docserver.start()
            except Exception as e:
                self.error = e

        def ready(self, server):
            self.serving = True
            self.host = server.host
            self.port = server.server_port
            self.url = 'http://%s:%d/' % (self.host, self.port)

        def stop(self):
            """Stop the server and this thread nicely."""
            self.docserver.close()
            self.join()
            # explicitly break a reference cycle: DocServer.callback
            # has indirectly a reference to ServerThread.
            self.docserver = None
            self.serving = False
            self.url = None

    thread = ServerThread(hostname, port)
    thread.daemon = True
    thread.start()
    # Wait until thread.serving is True to make sure we are
    # really up before returning.
    while not thread.error and not thread.serving:
        time.sleep(.01)
    return thread
